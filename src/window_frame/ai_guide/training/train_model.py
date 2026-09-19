"""Run from the project root: python -m window_frame.ai_guide.training.train_model."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from ..dataset.check_labels import audit_dataset, discover_samples
from ..dataset.split_dataset import split_dataset
from ..model.window_frame_model import WindowFrameModel
from ..evaluation.measure_accuracy import AccuracyMeter
from ..run_history.create_run_folder import create_run
from ..run_history.show_progress import Progress, format_duration
from ..utils.save_files import write_json, file_hash
from ..utils.setup_device import choose_device, seed_everything
from ..utils.check_settings import validate_config, check_resume
from ..utils.prepare_batches import to_device, limit_samples, create_loaders
from ..utils.save_examples import visualization_ids, save_comparisons
from ..visualization.plot_training_progress import open_dashboard, record_dashboard, plot_training_progress
from .calculate_loss import calculate_loss
from .configure_hardware import resource_plan, configure_hardware
from .check_early_stopping import EarlyStopping
from .estimate_remaining_time import RemainingTime
from .save_and_load import save_checkpoint, copy_checkpoint, load_checkpoint, restore_random_state

from ..utils.settings import settings


def run_training(config, resume=None, audit_only=False):
    validate_config(config)
    if not audit_only and not config.dataset.class_meanings_confirmed:
        raise ValueError("Rail label meanings need confirmation. Use --confirm-labels after checking the mapping, "
                         "or set dataset.class_meanings_confirmed=true in your config.")
    run, logger = create_run("training", config.output.root)
    write_json(run / "config.json", asdict(config))
    training, data = config.training, config.dataset
    device = choose_device(training.device)
    plan = resource_plan(training, device.type)
    logger.info("Available CPUs: %d; audit workers: %d", plan['available_cpus'], plan['audit_workers'])
    if training.skip_audit:
        logger.warning("Dataset audit skipped by training.skip_audit=true")
        report = dict(root=str(Path(data.root).resolve()), samples=discover_samples(data.root),
                      errors=[], skipped=True)
    else:
        with Progress("Checking dataset", logger) as progress:
            report = audit_dataset(data.root, workers=plan['audit_workers'], on_progress=progress.update)
        if report["errors"]:
            write_json(run / "dataset_audit.json", report)
            raise ValueError(f"Dataset audit failed; see {run / 'dataset_audit.json'}")
        logger.info("Audited %d samples, %d empty masks", len(report["samples"]), report["empty_masks"])
    write_json(run / "dataset_audit.json", report)
    if audit_only:
        return run
    hardware = configure_hardware(training, device, plan)
    write_json(run / "hardware.json", hardware)
    logger.info("Training device: %s; compute threads: %d; data workers: %d; GPU: %s",
                device, plan['cpu_threads'], plan['workers'], hardware['gpu_name'])
    seed_everything(training.seed)
    generator = torch.Generator().manual_seed(training.seed)
    groups = json.loads(Path(data.scene_groups).read_text()) if data.scene_groups else None
    state = None
    if resume:
        model, state = load_checkpoint(resume, device)
        check_resume(config, state, resume, report["samples"])
        split = state["split"]
        write_json(run / "resumed_from.json", dict(path=str(Path(resume).resolve()), sha256=file_hash(resume)))
    else:
        model = WindowFrameModel(base_channels=config.model.base_channels,
                                 max_correction=config.model.max_correction,
                                 confidence=config.model.confidence).to(device)
        split = split_dataset(report["samples"], training.seed, data.validation_fraction, data.test_fraction, groups)
        split = limit_samples(split, training)
    logger.info('Using %d training, %d validation, %d held-out test examples',
                len(split['train']), len(split['validation']), len(split['test']))
    if hardware['channels_last']:
        model = model.to(memory_format=torch.channels_last)
    write_json(run / "dataset_split.json", split)
    if groups is None:
        logger.warning(split["warning"])
    loaders = create_loaders(config, split, plan, device, generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    # Each new run selects its own best; the parent run retains its checkpoints.
    best, start = math.inf, 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        # Honor the new run's explicitly requested optimizer hyperparameters.
        for group in optimizer.param_groups:
            group.update(lr=training.learning_rate, weight_decay=training.weight_decay)
        start = state["epoch"] + 1
        restore_random_state(state, generator)
    if start >= training.epochs:
        raise ValueError("epochs must exceed the resumed checkpoint epoch")
    stopping = EarlyStopping(training.early_stopping_patience,
                             training.early_stopping_min_delta,
                             training.segmentation_epochs + training.enhancement_epochs)
    if state and stopping.restore(state.get('early_stopping')):
        logger.info('Restored early-stopping counter: %d/%d', stopping.bad_epochs, stopping.patience)
        if stopping.patience and stopping.bad_epochs >= stopping.patience:
            raise ValueError('This checkpoint already stopped early. Change early_stopping_patience '
                             '(or set it to 0) to deliberately continue.')
    if stopping.patience:
        logger.info("Early stopping: patience %d, minimum improvement %s, monitoring from epoch %d",
                    stopping.patience, stopping.min_delta, stopping.start_epoch + 1)
    else:
        logger.info("Early stopping: disabled")
    remaining_time = RemainingTime(training.epochs, len(loaders['train']) + len(loaders['validation']))
    stop_reason = 'maximum_epochs'
    dashboard = open_dashboard(run / "dashboard", training.tensorboard)
    history = []
    (run / "visualizations").mkdir()
    amp = training.mixed_precision and device.type == "cuda" and torch.cuda.is_bf16_supported()
    logger.info("Device: %s; BF16: %s; parameters: %d", device, amp, sum(p.numel() for p in model.parameters()))
    accumulation = training.accumulation_steps
    progress = None
    try:
        for epoch in range(start, training.epochs):
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if epoch < training.segmentation_epochs:
                stage = "segmentation"
            elif epoch < training.segmentation_epochs + training.enhancement_epochs:
                stage = "enhancement"
            else:
                stage = "joint"
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, examples = 0.0, 0
            progress = Progress(f"Epoch {epoch+1}/{training.epochs} training ({stage})", logger, len(loaders['train']))
            for index, batch in enumerate(loaders["train"]):
                batch = to_device(batch, device)
                if hardware['channels_last']:
                    batch['image'] = batch['image'].contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    result = model(batch["image"])
                    loss = calculate_loss(result, batch, stage)["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                group_start = (index // accumulation) * accumulation
                group_length = min(accumulation, len(loaders["train"]) - group_start)
                (loss / group_length).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(loaders["train"]):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total += loss.item() * len(batch["image"])
                examples += len(batch["image"])
                progress.update(index+1, loss=total/examples,
                                remaining_seconds=remaining_time.estimate(epoch, time.perf_counter()-epoch_start, index+1))
            progress.close()
            training_seconds = time.perf_counter() - epoch_start
            model.eval()
            meter = AccuracyMeter()
            val_total, val_examples = 0.0, 0
            selected_ids = visualization_ids(split['validation'], epoch, training.seed)
            progress = Progress(f"Epoch {epoch+1}/{training.epochs} validation", logger, len(loaders['validation']))
            with torch.inference_mode():
                for index, batch in enumerate(loaders["validation"]):
                    batch = to_device(batch, device)
                    result = model(batch["image"])
                    # Fixed validation objective makes checkpoints comparable across stages.
                    loss = calculate_loss(result, batch, "joint")["total"]
                    val_total += loss.item() * len(batch["image"])
                    val_examples += len(batch["image"])
                    meter.update(result, batch)
                    progress.update(index+1, loss=val_total/val_examples,
                                    remaining_seconds=remaining_time.estimate(epoch, time.perf_counter()-epoch_start,
                                                                              len(loaders['train'])+index+1))
                    save_comparisons(batch, result, selected_ids, run / "visualizations", epoch, dashboard)
            progress.close()
            metrics = meter.compute()
            row = dict(epoch=epoch+1, stage=stage, train_loss=total/examples,
                       validation_loss=val_total/val_examples, seconds=time.perf_counter()-epoch_start,
                       training_images_per_second=examples / max(training_seconds, 1e-9),
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       metrics=metrics)
            if not math.isfinite(row["validation_loss"]):
                raise FloatingPointError("Non-finite validation loss")
            should_stop = stopping.update(epoch, row['validation_loss'])
            row['early_stopping'] = stopping.state_dict()
            history.append(row)
            plot_training_progress(history, run / "loss.svg")
            record_dashboard(dashboard, epoch+1, row)
            # Explicit selection rule: fixed joint validation loss, not test performance.
            score = row["validation_loss"]
            improved = score < best
            best = min(best, score)
            save_checkpoint(run / "checkpoints" / "last.pt", model, optimizer, epoch, best, config, split, metrics, generator,
                            early_stopping=row['early_stopping'])
            if improved:
                copy_checkpoint(run / "checkpoints" / "last.pt", run / "checkpoints" / "best.pt")
            duration = time.perf_counter() - epoch_start
            average_duration = remaining_time.finish_epoch(duration)
            eta = 0 if should_stop else average_duration * (training.epochs - epoch - 1)
            row['seconds'] = duration
            row['estimated_remaining_seconds'] = eta
            write_json(run / 'history.json', history)
            logger.info("Epoch %d/%d [%s]: train %.4f validation %.4f | all epochs remaining ~%s",
                        epoch+1, training.epochs, stage, row["train_loss"], score,
                        format_duration(eta))
            if stopping.patience and epoch >= stopping.start_epoch:
                logger.info('Early stopping: %d/%d epochs without sufficient validation improvement',
                            stopping.bad_epochs, stopping.patience)
            if should_stop:
                stop_reason = 'early_stopping'
                logger.info('Stopped early at epoch %d. Best and latest checkpoints have been saved.', epoch+1)
                break
        write_json(run / 'summary.json', dict(stop_reason=stop_reason, completed_epoch=epoch+1,
                   planned_epochs=training.epochs, best_validation_loss=best,
                   early_stopping=stopping.state_dict()))
    except BaseException:
        if progress:
            progress.close('stopped')
        logger.exception("Training stopped; last.pt contains the last completed epoch, if any")
        raise
    finally:
        if dashboard:
            dashboard.close()
    return run


def main():
    parser = argparse.ArgumentParser(description="Train your window-frame model from scratch")
    parser.add_argument("--resume", type=Path)
    parser.add_argument('--epochs', type=int, help='Override the total epoch count')
    parser.add_argument('--max-train-samples', type=int, help='Use at most this many training examples')
    parser.add_argument('--max-validation-samples', type=int, help='Use at most this many validation examples')
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--confirm-labels", action="store_true",
                        help="Confirm IDs: 0 background, 1 clean neutral, 2 neutral, 3 wood, 4 dark, 5 protected")
    args = parser.parse_args()
    config = settings
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.max_train_samples is not None:
        config.training.max_train_samples = args.max_train_samples
    if args.max_validation_samples is not None:
        config.training.max_validation_samples = args.max_validation_samples
    if args.confirm_labels:
        config.dataset.class_meanings_confirmed = True
    try:
        run_training(config, args.resume, args.audit_only)
    except ValueError as error:
        parser.exit(2, f"Training setup error: {error}\n")


if __name__ == "__main__":
    main()
