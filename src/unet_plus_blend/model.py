"""Variable-count exposure blending with a nested U-Net++ feature extractor."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )


class UNetPlusBlend(nn.Module):
    """U-Net++ blend accepting [B, N, 3, H, W], returning [B, 3, H, W].

    Each exposure passes through the same nested U-Net++. At node X[i,j],
    concatenate X[i,0] through X[i,j-1] with upsampled X[i+1,j-1].
    This gives dense nested skip paths rather than ordinary U-Net skips.

    Final per-image features and their set mean determine exposure scores.
    Softmax across images produces RGB-shared per-pixel blending weights.
    No deep supervision or pretrained encoder is used.
    """

    def __init__(self, base_channels: int = 16, depth: int = 3):
        super().__init__()
        if base_channels < 1 or depth < 1:
            raise ValueError("base_channels and depth must be positive")
        self.base_channels = base_channels
        self.depth = depth
        channels = [base_channels * 2**level for level in range(depth + 1)]
        self.pool = nn.MaxPool2d(2)
        self.nodes = nn.ModuleDict()
        for level in range(depth + 1):
            incoming = 3 if level == 0 else channels[level - 1]
            self.nodes[f"{level}_0"] = DoubleConv(incoming, channels[level])
        for column in range(1, depth + 1):
            for level in range(depth - column + 1):
                incoming = column * channels[level] + channels[level + 1]
                self.nodes[f"{level}_{column}"] = DoubleConv(incoming, channels[level])
        self.score = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("Expected images shaped [batch, count, 3, height, width]")
        batch, count, _, height, width = images.shape
        if batch < 1 or count < 1:
            raise ValueError("Batch and image count must be positive")
        if min(height, width) < 2**self.depth:
            raise ValueError(f"Height and width must be at least {2**self.depth}")
        if not images.is_floating_point():
            raise ValueError("Use floating-point RGB images normalized to [0, 1]")

        inputs = images.reshape(batch * count, 3, height, width)
        features = {(0, 0): self.nodes["0_0"](inputs)}
        for level in range(1, self.depth + 1):
            features[level, 0] = self.nodes[f"{level}_0"](
                self.pool(features[level - 1, 0])
            )
        for column in range(1, self.depth + 1):
            for level in range(self.depth - column + 1):
                upsampled = F.interpolate(
                    features[level + 1, column - 1],
                    size=features[level, 0].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                skips = [features[level, previous] for previous in range(column)]
                features[level, column] = self.nodes[f"{level}_{column}"](
                    torch.cat([*skips, upsampled], dim=1)
                )

        per_image = features[0, self.depth].reshape(
            batch, count, self.base_channels, height, width
        )
        context = per_image.mean(dim=1, keepdim=True).expand_as(per_image)
        scoring_inputs = torch.cat((per_image, context), dim=2)
        logits = self.score(scoring_inputs.reshape(
            batch * count, self.base_channels * 2, height, width
        )).reshape(batch, count, 1, height, width)
        weights = logits.softmax(dim=1)
        return (images * weights).sum(dim=1)

    def blend(self, *images: Tensor) -> Tensor:
        """Blend any positive number of matching [B, 3, H, W] tensors."""
        if not images:
            raise ValueError("Provide at least one image")
        first = images[0]
        if first.ndim != 4 or first.shape[1] != 3:
            raise ValueError("Each image must be shaped [batch, 3, height, width]")
        if any(
            image.shape != first.shape
            or image.dtype != first.dtype
            or image.device != first.device
            for image in images
        ):
            raise ValueError("Images must have matching shapes, dtypes and devices")
        return self(torch.stack(images, dim=1))
