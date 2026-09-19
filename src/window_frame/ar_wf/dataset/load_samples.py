from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset


def input_patch(folder: Path):
    matches = [
        folder / name
        for name in ("input.png", "input.jpg", "input.jpeg")
        if (folder / name).is_file()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one input image in {folder}, found {len(matches)}"
        )

    return matches[0]


