import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two convolutions that preseve the height and width"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class ArNet(nn.Module):
    """A two-level encoder/decoder with skip connections."""

    def __init__(self, base_channels=8):
        super().__init__()

        self.enconder1 = DoubleConv(9, base_channels)
        self.enconder2 = DoubleConv(base_channels, base_channels * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)

        self.decoder2 = DoubleConv(
            base_channels * 4 + base_channels * 2, base_channels * 2
        )
        self.decoder1 = DoubleConv(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, dark, middle, bright):
        if dark.ndim != 4 or dark.shape[1] != 3:
            raise ValueError("Expected input shaped [batch, 3, height, width]")
        if dark.shape != middle.shape or dark.shape != bright.shape:
            raise ValueError("All three exposures must have matching shapes")
        if min(dark.shape[-2:]) < 4:
            raise ValueError("Height and width must both be at least 4")

        inputs = torch.cat([dark, middle, bright], dim=1)

        skip1 = self.enconder1(inputs)
        skip2 = self.enconder2(self.pool(skip1))
        features = self.bottleneck(self.pool(skip2))

        features = F.interpolate(
            features, size=skip2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder2(torch.cat([features, skip2], dim=1))
        features = F.interpolate(
            features, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder1(torch.cat([features, skip1], dim=1))

        weights = self.head(features).softmax(dim=1)
        exposures = torch.stack([dark, middle, bright], dim=1)
        return (exposures * weights.unsqueeze(2)).sum(dim=1)
