"""Photo-only inference; restore output dimensions before protected composition."""
import argparse
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from PIL import Image, ImageOps
from ..dataset.prepare_images import image_tensor
from ..model.window_frame_model import compose
from ..training.save_and_load import load_checkpoint
from ..run_history.create_run_folder import create_run, DEFAULT_OUTPUT
from ..run_history.record_run_details import choose_device, write_json, file_hash
from .save_results import save_results
from .process_large_image import process_large_image


@torch.inference_mode()
def predict_image(model, image, size, device, mode="tiled", overlap=None):
    """Return original-size CPU outputs; training size controls crop size only."""
    if size < 16:
        raise ValueError("Prediction size must be at least 16")
    original = image_tensor(image.convert("RGB")).unsqueeze(0)
    if mode == "tiled":
        overlap = size // 4 if overlap is None else overlap
        probabilities, soft_mask, residual = process_large_image(model, original, size, overlap, device)
    elif mode == "resized":
        resized = image_tensor(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).unsqueeze(0).to(device)
        prediction = model(resized)
        dimensions = original.shape[-2:]
        resize = lambda x: F.interpolate(x.float().cpu(), size=dimensions, mode="bilinear", align_corners=False)
        probabilities = resize(prediction["class_logits"]).softmax(1)
        soft_mask = resize(prediction["mask_logits"]).sigmoid()
        residual = resize(prediction["residual"])
    else:
        raise ValueError("mode must be 'tiled' or 'resized'")
    enhanced, gate = compose(original, residual, probabilities, soft_mask, model.settings["confidence"])
    return dict(enhanced=enhanced, edit_gate=gate, probabilities=probabilities, soft_mask=soft_mask)


def run_prediction(checkpoint, source, output_root=DEFAULT_OUTPUT, device="auto", confidence=None,
                   mode="tiled", overlap=None):
    device = choose_device(device)
    model, state = load_checkpoint(checkpoint, device)
    model.eval()
    if confidence is not None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        model.settings["confidence"] = confidence
    source = Path(source)
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if source.is_dir() else [source]
    if not paths:
        raise ValueError("No supported images found")
    run, logger = create_run("predictions", output_root)
    identity = dict(path=str(Path(checkpoint).resolve()), sha256=file_hash(checkpoint), epoch=state["epoch"]+1)
    write_json(run / "settings.json", dict(checkpoint=identity, device=str(device), model=model.settings,
        tile_size=state["config"]["dataset"]["image_size"], prediction_mode=mode,
        overlap=(state["config"]["dataset"]["image_size"] // 4 if overlap is None else overlap) if mode == "tiled" else None))
    successful, failures = [], []
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            size = state["config"]["dataset"]["image_size"]
            result = predict_image(model, image, size, device, mode, overlap)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            folder = run / f"{index:05d}_{path.stem}"
            save_results(folder, image, result, dict(source=str(path.resolve()), source_sha256=file_hash(path),
                checkpoint=identity, inference_seconds=seconds, device=str(device), width=image.width,
                height=image.height, confidence_threshold=model.settings["confidence"],
                prediction_mode=mode, tile_size=size,
                overlap=(size // 4 if overlap is None else overlap) if mode == "tiled" else None))
            successful.append(folder.name)
            logger.info("Saved %s (%.3fs)", folder.name, seconds)
        except (OSError, ValueError) as error:
            failures.append(dict(source=str(path), error=str(error)))
            logger.exception("Could not process %s", path)
    write_json(run / "summary.json", dict(successful=successful, failures=failures, checkpoint=identity))
    if failures:
        raise RuntimeError(f"{len(failures)} image(s) failed; see {run / 'summary.json'}")
    return run


def main():
    parser = argparse.ArgumentParser(description="Enhance a photo or folder and export masks, guides, and metadata")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--mode", choices=["tiled", "resized"], default="tiled",
                        help="tiled keeps native detail; resized is the old, faster whole-photo mode")
    parser.add_argument("--overlap", type=int, help="Shared pixels between crops; default: quarter of training size")
    args = parser.parse_args()
    run_prediction(args.checkpoint, args.input, args.output_root, args.device, args.confidence,
                   args.mode, args.overlap)


if __name__ == "__main__":
    main()
