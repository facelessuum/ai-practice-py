"""Read paired examples without changing source data."""
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from .prepare_images import prepare_sample


def input_path(folder):
    matches = [folder / name for name in ("input.png", "input.jpg", "input.jpeg") if (folder / name).is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one input image in {folder}, found {len(matches)}")
    return matches[0]


class WindowFrameDataset(Dataset):
    def __init__(self, root, sample_ids, size=512, augment=False):
        self.root, self.ids = Path(root), list(sample_ids)
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        folder = self.root / name
        images = []
        for path, mode in [(input_path(folder), "RGB"), (folder / "output.png", "RGB"),
                           (folder / "mask.png", "L"), (folder / "rail_classes.png", None)]:
            with Image.open(path) as image:
                images.append(image.convert(mode) if mode else image.copy())
        sample = prepare_sample(*images, self.size, self.augment)
        classes = sample["classes"]
        if not torch.all((classes <= 5) | (classes == 255)):
            raise ValueError(f"Invalid class ID in {name}")
        # Preserve frame supervision but don't call unlabeled frame pixels background.
        unknown = (sample["mask"][0] > 0) & (classes == 0)
        sample["classes"] = classes.masked_fill(unknown, 255)
        sample["id"] = name
        return sample
