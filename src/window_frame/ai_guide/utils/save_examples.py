"""Select and save validation examples during training."""
import random

import numpy as np

from ..visualization.compare_images import compare_images, to_image


def visualization_ids(sample_ids, epoch, seed):
    """Rotate through a reproducible order without changing training randomness."""
    ids = list(sample_ids)
    random.Random(seed).shuffle(ids)
    count = min(5, len(ids))
    return {ids[(epoch * count + i) % len(ids)] for i in range(count)}


def save_comparisons(batch, result, selected_ids, folder, epoch, dashboard):
    for index, sample_id in enumerate(batch["id"]):
        if sample_id not in selected_ids:
            continue
        images = {
            "Input": to_image(batch["image"][index]),
            "Target": to_image(batch["target"][index]),
            "Enhanced": to_image(result["enhanced"][index]),
            "True mask": to_image(batch["mask"][index]),
            "Soft mask": to_image(result["soft_mask"][index]),
        }
        comparison = compare_images(images, folder / f"epoch_{epoch+1:04d}_{sample_id}.png")
        if dashboard:
            dashboard.add_image(
                f"validation/comparison/{sample_id}", np.asarray(comparison), epoch+1, dataformats="HWC",
            )
