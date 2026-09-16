"""Load three aligned exposures and their supervised target image."""

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_inputs(case):
    # case = Path("data/cases/arwJob1_g27_5060ti_zed")
    """Load the three exposures and the matching supervised target."""
    paths = sorted((case / "10_converted").glob("*.jpg"))
    target_path = case / "50_ai_blend" / "output.jpg"
    if len(paths) != 3:
        raise ValueError(f"{case}: expected exactly three aligned JPEGs")
    if not target_path.is_file():
        raise ValueError(f"{case}: missing supervised target {target_path}")

    with Image.open(paths[0]) as image:
        original_size = image.size
    for path in paths[1:]:
        with Image.open(path) as image:
            if image.size != original_size:
                raise ValueError(f"{case}: aligned images must have matching dimensions")

    new_size = (
        max(1, round(original_size[0] * 0.15)),
        max(1, round(original_size[1] * 0.15)),
    )

    def load(path):
        with Image.open(path) as image:
            image = image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
            pixels = np.array(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()

    return load(paths[0]), load(paths[1]), load(paths[2]), load(target_path)


class ExposureDataset(Dataset):
    def __init__(self, root):
        self.cases = sorted(p for p in root.iterdir() if p.is_dir())
        if not self.cases:
            raise ValueError(f"No cases found in {root}")
        # Load one case on demand rather than caching every full-size image.

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        return load_inputs(self.cases[index])
