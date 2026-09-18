"""Evaluate saved held-out labels. Unlabeled test photos are for prediction only."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from ..dataset.load_samples import WindowFrameDataset
from ..dataset.check_labels import audit_dataset
from ..training.save_and_load import load_checkpoint
from ..training.train_model import to_device
from ..run_history.create_run_folder import create_run, DEFAULT_OUTPUT
from ..run_history.record_run_details import choose_device, write_json, file_hash
from ..visualization.compare_images import compare_images, to_image
from ..visualization.draw_rail_guide import draw_rail_guide
from ..visualization.plot_evaluation_results import plot_evaluation_results
from .measure_accuracy import AccuracyMeter


def run_evaluation(checkpoint, subset="test", output_root=DEFAULT_OUTPUT, device="auto", batch_size=4):
    device = choose_device(device)
    model, state = load_checkpoint(checkpoint, device)
    model.eval()
    data = state["config"]["dataset"]
    original_audit = Path(checkpoint).resolve().parent.parent / "dataset_audit.json"
    report = audit_dataset(data["root"])
    if report["errors"] or not original_audit.exists() or report["samples"] != json.loads(original_audit.read_text())["samples"]:
        raise ValueError("Dataset has changed, or original audit is missing; cannot reproduce evaluation")
    dataset = WindowFrameDataset(data["root"], state["split"][subset], data["image_size"])
    if not len(dataset):
        raise ValueError("Selected split is empty")
    run, logger = create_run("evaluation", output_root)
    (run / "visualizations").mkdir()
    meter = AccuracyMeter()
    individual = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = to_device(batch, device)
            result = model(batch["image"])
            meter.update(result, batch)
            for i, name in enumerate(batch["id"]):
                sample_meter = AccuracyMeter()
                sample_result = {k: v[i:i+1] for k, v in result.items()}
                sample_batch = {k: v[i:i+1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                sample_meter.update(sample_result, sample_batch)
                individual.append(dict(id=name, metrics=sample_meter.compute()))
                if len(individual) <= 24:
                    original = to_image(batch["image"][i])
                    labels = result["probabilities"][i].argmax(0).cpu().numpy()
                    compare_images({"Input": original, "Target": to_image(batch["target"][i]),
                        "Enhanced": to_image(result["enhanced"][i]), "Soft mask": to_image(result["soft_mask"][i]),
                        "Rail guide": draw_rail_guide(original, labels)}, run / "visualizations" / f"{name}.png")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    summary = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=file_hash(checkpoint),
        epoch=state["epoch"]+1, subset=subset, samples=len(dataset), metrics=meter.compute(),
        elapsed_seconds=seconds, timing_note="Includes data loading, metrics, and visualization; not pure inference speed.",
        device=str(device), peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        image_size=data["image_size"], confidence_threshold=model.settings["confidence"],
        grouping=state["split"]["grouping"])
    write_json(run / "summary.json", summary)
    plot_evaluation_results(summary["metrics"], run / "evaluation.svg")
    write_json(run / "per_image.json", individual)
    logger.info("Evaluation complete: %s", run / "summary.json")
    return run


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trusted checkpoint on its saved held-out split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.split, args.output_root, args.device, args.batch_size)


if __name__ == "__main__":
    main()
