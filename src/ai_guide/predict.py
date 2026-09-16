"""Combine a case's three aligned exposures using saved model weights."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .dataset import load_inputs
from .model import BlendModel
from .unet import UNetBlendModel


def load_model(path):
    """Load trained weights once for one or many predictions."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Older tiny-CNN checkpoints have no architecture field.
    architecture = checkpoint.get("architecture", "blend")
    if architecture == "unet":
        model = UNetBlendModel(base_channels=checkpoint["base_channels"])
    elif architecture == "blend":
        model = BlendModel()
    else:
        raise ValueError(f"Unknown checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return model


def blend_case(model, case, destination):
    """Blend full-resolution aligned inputs without resizing or tiling."""
    dark, middle, bright, _ = load_inputs(case)
    with torch.inference_mode():
        batch = model(
            dark.unsqueeze(0),
            middle.unsqueeze(0),
            bright.unsqueeze(0),
        )
        output = batch[0]
    pixels = (
        (output.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(destination)
 

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("runs/ai_guide/unet.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/ai_guide/blend.png"))
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path("data").resolve()) or args.output.resolve().is_relative_to(args.case.resolve()):
        parser.error("Save predictions outside the input data")
    torch.set_num_threads(4)
    model = load_model(args.model)
    blend_case(model, args.case, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
