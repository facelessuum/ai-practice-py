"""Order-independent U-Net exposure fusion with a variable image count."""

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


class DynamicUNetBlend(nn.Module):
    """Blend [B, N, 3, H, W] exposures into [B, 3, H, W], for N >= 1.

    A shared U-Net extracts features from each exposure. Each exposure's
    score depends on its features and the mean features of the whole set.
    Softmax across exposures produces per-pixel blending weights.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        c = base_channels
        self.encoder1 = DoubleConv(3, c)
        self.encoder2 = DoubleConv(c, c * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 2, c * 4)
        self.decoder2 = DoubleConv(c * 6, c * 2)
        self.decoder1 = DoubleConv(c * 3, c)
        self.score = nn.Sequential(
            nn.Conv2d(c * 2, c, 1),
            nn.ReLU(),
            nn.Conv2d(c, 1, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("Expected images shaped [batch, count, 3, height, width]")
        batch, count, _, height, width = images.shape
        if batch < 1 or count < 1:
            raise ValueError("Batch and image count must be positive")
        if min(height, width) < 4:
            raise ValueError("Height and width must be at least 4")
        if not images.is_floating_point():
            raise ValueError("Use floating-point RGB images normalized to [0, 1]")

        # Process all exposures with the same parameters, not N-specific layers.
        inputs = images.reshape(batch * count, 3, height, width)
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(self.pool(skip1))
        features = self.bottleneck(self.pool(skip2))
        features = F.interpolate(
            features, size=skip2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder2(torch.cat((features, skip2), dim=1))
        features = F.interpolate(
            features, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder1(torch.cat((features, skip1), dim=1))

        channels = features.shape[1]
        features = features.reshape(batch, count, channels, height, width)
        context = features.mean(dim=1, keepdim=True).expand_as(features)
        scoring_inputs = torch.cat((features, context), dim=2)
        logits = self.score(
            scoring_inputs.reshape(batch * count, channels * 2, height, width)
        ).reshape(batch, count, 1, height, width)
        weights = logits.softmax(dim=1)
        return (images * weights).sum(dim=1)

    def blend(self, *images: Tensor) -> Tensor:
        """Convenience interface: model.blend(image1, image2, ...).

        Each image must be [B, 3, H, W] with matching shape, dtype and device.
        """
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
