"""Variable-count exposure loading; targets are required only for training."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def image_tensor(path: Path, size: tuple[int, int]) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        pixels = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def load_inputs(case: Path, scale: float = 0.1, min_size: int = 8) -> Tensor:
    """Return [N, 3, H, W]. Inputs must already be spatially aligned."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    paths = sorted(
        path for path in (case / "10_converted").glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No input images in {case / '10_converted'}")
    original_size = None
    for path in paths:
        with Image.open(path) as image:
            if original_size is None:
                original_size = image.size
            elif image.size != original_size:
                raise ValueError(f"Input dimensions differ in {case}")
    width, height = original_size
    size = (round(width * scale), round(height * scale))
    if min(size) < min_size:
        raise ValueError(f"Scaled dimensions {size} are too small in {case}; increase scale")
    return torch.stack([image_tensor(path, size) for path in paths])


class ExposureDataset(Dataset):
    """Each item is (exposures [N,3,H,W], target [3,H,W])."""

    def __init__(self, root: Path, scale: float = 0.1, min_size: int = 8):
        self.scale = scale
        self.min_size = min_size
        self.cases = []
        self.skipped_cases = []
        for case in sorted(root.iterdir()):
            if not case.is_dir():
                continue
            if (case / "50_ai_blend" / "output.jpg").is_file():
                self.cases.append(case)
            else:
                self.skipped_cases.append(case)
        if not self.cases:
            raise ValueError(f"No cases with 50_ai_blend/output.jpg targets in {root}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        inputs = load_inputs(case, self.scale, self.min_size)
        target = image_tensor(
            case / "50_ai_blend" / "output.jpg",
            (inputs.shape[-1], inputs.shape[-2]),
        )
        return inputs, target
