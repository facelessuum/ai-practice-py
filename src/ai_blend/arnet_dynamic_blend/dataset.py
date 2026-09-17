from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
from PIL import Image
import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def image_tensor(path: Path, size: tuple[int, int]) -> Tensor:

    # Open the image in this path and name it as image variable
    with Image.open(path) as image:
        # convert the image into RGB and each pixel will became an array of [R,G,B]
        # we resize the image base on the size arguments and we select this Resampling.LANCZOS because I have no idea what it does
        resized_image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        # we put the array of RGB's into numpy array so it'll be more effecient
        # and then we devide it by 255.0 and convert it into a float32 to get more accurate RGB numbers
        pixels = np.array(resized_image, dtype=np.float32) / 255.0

    # we convert the pixels of RGB's into something torch can read
    # and then we switch the order so CNN can understand it
    # from [height, width, channels] -> [channels, height, width]
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


# needs a case Path and a scale size percentage and returns a tensor tuple
def load_inputs(case: Path, scale: float = 0.1) -> Tensor:
    """Return [N, 3, H, W]. Inputs must already be spatially aligned."""

    # we throw an error if scale isn't in between 0-1 (0-100%)
    if not 0 < scale <= 1:
        raise ValueError("Scale must be in (0,1)")
    paths = sorted(
        path
        for path in (case / "10_converted").glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not paths:
        raise ValueError(f"No input images in {case / '10_convereted'}")
    original_size = None

    for path in paths:
        with Image.open(path) as image:
            if original_size is None:
                original_size = image.size
            elif image.size != original_size:
                raise ValueError(f"Input dimensino differ in {case}")
    width, height = original_size
    size = (max(1, round(width * scale)), max(1, round(height * scale)))

    if min(size) < 4:
        raise ValueError(
            f"Scaled dimension {size} are too small in {case}; increase scale"
        )

    return torch.stack([image_tensor(path, size) for path in paths])


class ExposureDataset(Dataset):
    """Each item is (exposures [N,3,H,W], target [3,H,W])."""

    def __init__(self, path: Path, scale: float = 0.1):
        self.scale = scale
        self.cases = []
        self.skipped_cases = []

        for case in sorted(path.iterdir()):
            if not case.is_dir():
                continue
            if (case / "50_ai_blend" / "output.jpg").is_file():
                self.cases.append(case)
            else:
                self.skipped_cases.append(case)

        if not self.cases:
            raise ValueError(f"No cases with 50_ai_blend/output.jpg targets in {path}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        inputs = load_inputs(case, self.scale)
        target = image_tensor(
            case / "50_ai_blend" / "output.jpg", (inputs.shape[-1], inputs.shape[-2])
        )

        return inputs, target
