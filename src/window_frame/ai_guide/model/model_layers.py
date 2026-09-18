"""Small building blocks, initialized from scratch."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.SiLU(),
        )


class Decoder(nn.Module):
    def __init__(self, widths: list[int]):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(widths[i + 1] + widths[i], widths[i])
            for i in reversed(range(len(widths) - 1))
        ])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return x
