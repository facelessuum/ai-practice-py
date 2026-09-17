"""Single-case or batch inference without loading target images."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .dataset import load_inputs
from .model import DynamicUNetBlend
from .utils import check_output, create_run, latest_checkpoint


def load_model(path: Path, device: str = "cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "dynamic_unet_blend":
        raise ValueError("Expected a dynamic_unet_blend checkpoint")
    model = DynamicUNetBlend(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["weights"])
    model.to(device).eval()
    return model, checkpoint["scale"]


def blend_case(model, case: Path, destination: Path, scale: float = 0.1):
    check_output(destination, case)
    images = load_inputs(case, scale).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model(images)[0]
    pixels = (
        (output.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", type=Path, help="Predict one case")
    source.add_argument("--data", type=Path, help="Predict every case directory")
    parser.add_argument("--model", type=Path, help="Checkpoint; defaults to latest run")
    parser.add_argument(
        "--model-root", type=Path, default=Path("runs/unet_dynamic_blend")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/unet_dynamic_blend")
    )
    parser.add_argument(
        "--scale", type=float, help="Default: checkpoint training scale"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or (args.scale is not None and not 0 < args.scale <= 1):
        parser.error("threads must be positive and scale must be in (0, 1]")
    root = args.case if args.case is not None else args.data
    if not root.is_dir():
        parser.error(f"Input directory does not exist: {root}")
    check_output(args.output, root)
    torch.set_num_threads(args.threads)
    path = args.model if args.model is not None else latest_checkpoint(args.model_root)
    model, saved_scale = load_model(path, args.device)
    scale = saved_scale if args.scale is None else args.scale
    cases = (
        [args.case]
        if args.case is not None
        else sorted(case for case in args.data.iterdir() if case.is_dir())
    )
    if not cases:
        parser.error("No case directories found")
    run = create_run(args.output)
    print(f"Using: {path}")
    for case in cases:
        destination = (
            run if args.case is not None else run / case.name
        ) / "ai_blend.jpg"
        blend_case(model, case, destination, scale)
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
