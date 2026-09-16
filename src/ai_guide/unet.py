"""A small U-Net that blends three aligned RGB exposures.

Inputs: dark, middle, bright, each shaped [batch, 3, height, width]
        with floating-point RGB values in [0, 1].
Output: a blended RGB image with the same shape and resolution.

This predicts exposure weights, not arbitrary edited colors. It cannot
recover detail missing from all three inputs or perform alignment itself.
train.py trains this model; its weights are not interchangeable with the tiny CNN.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two convolutions that preserve the height and width."""

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


class UNetBlendModel(nn.Module):
    """A two-level encoder/decoder with skip connections."""

    def __init__(self, base_channels=16):
        super().__init__()
        # Three exposures × three RGB channels = nine input channels.
        self.encoder1 = DoubleConv(9, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)

        # Each decoder receives upsampled features AND encoder features.
        self.decoder2 = DoubleConv(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = DoubleConv(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, dark, middle, bright):
        if dark.ndim != 4 or dark.shape[1] != 3:
            raise ValueError("Expected inputs shaped [batch, 3, height, width]")
        if dark.shape != middle.shape or dark.shape != bright.shape:
            raise ValueError("All three exposures must have matching shapes")
        if min(dark.shape[-2:]) < 4:
            raise ValueError("Height and width must both be at least 4")

        inputs = torch.cat([dark, middle, bright], dim=1)

        # Encoder: shrink spatial dimensions and learn broader context.
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(self.pool(skip1))
        features = self.bottleneck(self.pool(skip2))

        # Decoder: restore resolution, reusing fine detail via skip connections.
        # Explicit sizes also support odd image dimensions.
        features = F.interpolate(
            features, size=skip2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder2(torch.cat([features, skip2], dim=1))
        features = F.interpolate(
            features, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.decoder1(torch.cat([features, skip1], dim=1))

        # Three nonnegative weights per pixel, summing to one.
        weights = self.head(features).softmax(dim=1)
        exposures = torch.stack([dark, middle, bright], dim=1)
        return (exposures * weights.unsqueeze(2)).sum(dim=1)


if __name__ == "__main__":
    # Small synthetic example; random weights do not produce a trained edit.
    torch.manual_seed(42)
    model = UNetBlendModel()
    exposures = [torch.rand(1, 3, 65, 97) for _ in range(3)]
    model.eval()
    with torch.no_grad():
        output = model(*exposures)
    print("Input shape:", exposures[0].shape)
    print("Output shape:", output.shape)
    print("Output range:", output.min().item(), output.max().item())
