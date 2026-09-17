from torch import Tensor
from torch.nn import functional as TF

"""LOSS FUNC """
def blend_loss(output: Tensor, target: Tensor) -> Tensor:
    if output.shape != target.shape:
        raise ValueError(
            f"Prediction {output.shape} and target {target.shape} must match"
        )
    return TF.mse_loss(output, target)
