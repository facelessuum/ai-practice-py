"""Shared resizing and aligned augmentation. Never smooth class IDs."""
import random
import numpy as np
import torch
from PIL import Image, ImageOps


def image_tensor(image):
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1)


def prepare_sample(image, target, mask, classes, size, augment=False):
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    target = target.resize((size, size), Image.Resampling.BILINEAR)
    mask = mask.resize((size, size), Image.Resampling.BILINEAR)
    classes = classes.resize((size, size), Image.Resampling.NEAREST)
    if augment and random.random() < 0.5:
        image, target, mask, classes = [ImageOps.mirror(x) for x in (image, target, mask, classes)]
    return dict(image=image_tensor(image), target=image_tensor(target), mask=image_tensor(mask),
                classes=torch.from_numpy(np.asarray(classes, dtype=np.int64).copy()))
