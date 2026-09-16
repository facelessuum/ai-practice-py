from torch.utils.data import Dataset
from torch.fft import Tensor
import torch
from numpy.typing import NDArray
from numpy.core.multiarray import dtype
from PIL import Image
from pathlib import Path
import numpy as np

image_extensions = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "avif"}


def loadImage(path: Path, input_image: Tensor | None = None) -> Tensor:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        scale = 0.10
        if input_image is None:
            width, height = image.size
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
        else:
            size = input_image.shape[2], input_image.shape[1]
        resized_image = rgb_image.resize(size, Image.Resampling.LANCZOS)
        pixels = np.array(resized_image, dtype=np.float32) / 255.0
        tensor: Tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return tensor


def load_inputs(case: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    image_path = case / "10_converted"
    target_path = case / "50_ai_blend" / "output.jpg"

    image_input = sorted(
        path
        for path in image_path.glob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in image_extensions
    )

    images: list[Tensor] = [loadImage(image) for image in image_input]
    target = loadImage(target_path, images[0])

    return images[0], images[1], images[2], target


class ExposureDataset(Dataset):
    def __init__(self, root):
        self.cases = sorted(p for p in root.iterdir() if p.is_dir())
        if not self.cases:
            raise ValueError(f"No cases found in {root}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        return load_inputs(self.cases[index])




