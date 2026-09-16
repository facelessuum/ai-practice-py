"""Supervised RGB reconstruction objective."""

from torch import Tensor
from torch.nn import functional as F


def blend_loss(output: Tensor, target: Tensor) -> Tensor:
    if output.shape != target.shape:
        raise ValueError(f"Prediction {output.shape} and target {target.shape} must match")
    return F.mse_loss(output, target)
