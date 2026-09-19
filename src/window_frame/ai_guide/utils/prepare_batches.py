"""Select samples, build loaders, and move batches to the training device."""
import random

import torch
from torch.utils.data import DataLoader

from ..dataset.load_samples import WindowFrameDataset


def to_device(batch, device):
    moved = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device, non_blocking=device.type == "cuda")
        moved[name] = value
    return moved


def limit_samples(split, training):
    """Select reproducible small subsets after leakage-safe splitting."""
    split = dict(split)
    limits = {'train': training.max_train_samples, 'validation': training.max_validation_samples}
    for name, limit in limits.items():
        if limit is not None and limit < len(split[name]):
            split[name] = sorted(random.Random(training.seed).sample(split[name], limit))
    return split


def create_loaders(config, split, plan, device, generator):
    """Only the training loader shuffles and augments images."""
    worker_options = {}
    if plan["workers"]:
        worker_options = {"prefetch_factor": 2, "multiprocessing_context": "spawn"}
    loaders = {}
    for name in ("train", "validation"):
        is_training = name == "train"
        dataset = WindowFrameDataset(
            config.dataset.root, split[name], config.dataset.image_size, augment=is_training,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=is_training,
            num_workers=plan["workers"],
            pin_memory=device.type == "cuda",
            generator=generator,
            **worker_options,
        )
    return loaders
