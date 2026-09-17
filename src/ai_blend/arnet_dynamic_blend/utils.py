from utils.utils import DEVICE
from pathlib import Path
from .model import ArNetDynamicModel
import torch
"""Numbered runs and safeguards for read-only input data."""


def check_output(path: Path, input_root: Path) -> None:
    resolved = path.resolve()
    for protected in (Path("data").resolve(), input_root.resolve()):
        if resolved.is_relative_to(protected):
            raise ValueError(f"Save generated files outside input data: {path}")


def load_model():

    # Select the latest model
    model_path = max(
        Path("model").glob("arnet_blend/run_*/arnet_blend.pt"),
        key=lambda path: int(path.parent.name.removeprefix("run_")),
    )

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model = ArNetDynamicModel(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["weights"])
    model.to(DEVICE).eval()
    return model, checkpoint['scale']


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

def create_run(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    numbers = [
        int(path.name.removeprefix("run_"))
        for path in root.glob("run_*")
        if path.is_dir() and path.name.removeprefix("run_").isdigit()
    ]

    number = max(numbers, default=0) + 1

    while True:
        run = root / f"run_{number:04d}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            number += 1


def latest_checkpoint(root: Path) -> Path:
    paths = [
        path
        for path in root.glob("run_*/arnet_blend.pt")
        if path.is_file() and path.parent.name.removeprefix("run_").isdigit()
    ]
    if not paths:
        raise FileNotFoundError(f"No run_*/arnet_blend.pt checkpoints in {root}")

    return max(paths, key=lambda path: int(path.parent.name.removeprefix("run_")))
