from PIL import Image
from app.dataset import load_inputs
from pathlib import Path
import torch
from .model import ArNet
import numpy as np


def get_output_fold(case: str | None = None) -> Path:
    run_number = 1
    output_path = Path("output")
    while True:
        run_fold = output_path / f"run_{run_number:04d}"
        try:
            run_fold.mkdir(parents=True)
        except FileExistsError:
            run_number += 1
            continue

        if case is None:
            return run_fold / "ai_blend.jpg"

        return run_fold / case / "ai_blend.jpg"


def load_model():

    # Select the latest model
    model_path = max(
        Path("model").glob("run_*/arnet.pt"),
        key=lambda path: int(path.parent.name.removeprefix("run_")),
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = ArNet()

    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return model


def blend_case(model, case, destination=None):

    if destination is None:
        destination = get_output_fold()

    dark, middle, bright, _ = load_inputs(case)
    with torch.inference_mode():
        output = model(dark.unsqueeze(0), middle.unsqueeze(0), bright.unsqueeze(0))[0]
    pixels = (
        (output.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    )
    Image.fromarray(pixels).save(destination)


def blend_all():
    dataset = Path("data/cases")
    model = load_model()
    output_dir = get_output_fold().parent

    for case in dataset.iterdir():
        if case.is_dir():
            destination = output_dir / case.name / "ai_blend.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            blend_case(model, case, destination)
