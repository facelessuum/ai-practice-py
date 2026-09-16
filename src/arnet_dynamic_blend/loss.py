from torch import Tensor
from torch.nn import functional as TF

"""LOSS FUNC """
def blend_loss(output: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if output.shape != target.shape:
        raise ValueError(
            f"Prediction {output.shape} and target {target.shape} must match"
        )
    if mask is None:
        return TF.mse_loss(output, target)
    if mask.shape != (output.shape[0], 1, *output.shape[-2:]):
        raise ValueError("Mask must have shape [batch, 1, height, width]")
    # Mask is 1 for source pixels and 0 for padding, broadcast across RGB.
    error = (output - target).square() * mask
    return error.sum() / (mask.sum() * output.shape[1]).clamp_min(1)
