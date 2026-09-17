from torch.utils.data import DataLoader
from utils.utils import DEVICE
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from .utils import (
    create_run,
    latest_checkpoint,
    get_output_fold,
    load_model,
)
from .model import ArNetDynamicModel
from .dataset import load_inputs, ExposureDataset
import torch


def blend_single_case() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, default=Path("datasets/test/largeMovement_g3_5060ti_zed")
    )

    args = parser.parse_args()
    model, scale = load_model()

    destination = get_output_fold()

    images = (
        load_inputs(args.case, scale).unsqueeze(0).to(next(model.parameters()).device)
    )

    with torch.inference_mode():
        output = model(images)[0]
    pixels = (
        (output.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(destination)


def blend_all() -> None:
    dataset = Path("datasets/test")
    model, scale = load_model()
    output_dir = get_output_fold().parent

    for case in dataset.iterdir():
        if case.is_dir():
            destination = output_dir / case.name / "ai_blend.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            blend_case(model, case, destination, scale)


def blend_case(model, case: Path, destination: Path, scale: float = 0.1) -> None:

    if destination is None:
        destination = get_output_fold()
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
