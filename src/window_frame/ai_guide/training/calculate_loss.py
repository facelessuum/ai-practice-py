"""Balanced pixel supervision plus small, targeted image corrections."""
import torch
from torch.nn import functional as F


def region_mean(values, region):
    region = region.expand_as(values).to(values.dtype)
    return (values * region).sum() / region.sum().clamp_min(1)


def calculate_loss(result, batch, stage="joint"):
    image, target = batch["image"], batch["target"]
    classes, mask = batch["classes"], batch["mask"]
    valid = classes != 255
    ce = F.cross_entropy(result["class_logits"], classes, ignore_index=255, reduction="none")
    # Average observed classes equally, rather than allowing background to dominate.
    terms = [ce[classes == i].mean() for i in range(6) if (classes == i).any()]
    class_loss = torch.stack(terms).mean() if terms else result["class_logits"].sum() * 0
    bce = F.binary_cross_entropy_with_logits(result["mask_logits"], mask, reduction="none")
    foreground = mask > 0.05
    mask_loss = region_mean(bce, foreground) + region_mean(bce, ~foreground)
    p = result["soft_mask"]
    dims = (1, 2, 3)
    dice = (1 - (2 * (p * mask).sum(dims) + 1) / (p.sum(dims) + mask.sum(dims) + 1)).mean()
    eligible = ((classes >= 1) & (classes <= 4)).unsqueeze(1)
    if stage == "enhancement":
        gate = eligible.to(image.dtype) * mask
    else:
        # Soft predicted gate supplies gradients before strict inference gating opens.
        gate = result["probabilities"][:, 1:5].sum(1, keepdim=True).detach() * p.detach()
    candidate = (image + gate * result["residual"]).clamp(0, 1)
    reconstruction = region_mean((candidate - target).abs(), eligible)
    preserve = ((classes == 0) | (classes == 5)).unsqueeze(1)
    preservation = region_mean(result["residual"].abs(), preserve)
    total = class_loss + mask_loss + dice
    if stage != "segmentation":
        total = total + 5 * reconstruction + preservation
    return dict(total=total, classes=class_loss, mask=mask_loss, dice=dice,
                reconstruction=reconstruction, preservation=preservation)
