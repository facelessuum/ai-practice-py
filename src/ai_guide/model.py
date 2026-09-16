"""Learn how much of each of three aligned exposures to use at every pixel."""

import torch  # PyTorch provides tensor operations and automatic gradient computation.
from torch import nn  # nn provides neural-network layers.


class BlendModel(nn.Module):  # Inherit PyTorch's model features, such as tracking trainable parameters.
    def __init__(self):  # Build the layers when BlendModel() is created; return nothing.
        super().__init__()  # Initialize nn.Module before assigning any neural-network layers.
        self.network = nn.Sequential(  # Store layers that run in the order listed below.
            nn.Conv2d(9, 16, kernel_size=3, padding=1),  # Learn 16 feature maps from 9 channels (3 RGB photos), using 3x3 neighborhoods; padding preserves height/width.
            nn.ReLU(),  # Replace negative feature values with zero, adding a nonlinear operation.
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Turn 16 feature maps into 3 exposure scores per pixel—not 3 output colour channels.
        )  # Finish defining the sequence; its convolution weights will be learned during training.

    def forward(self, dark, middle, bright):  # Accept 3 image batches and return one blended image batch.
        # B = batch size, H = height, W = width. Each input is [B, 3, H, W], with RGB values in [0, 1].
        inputs = torch.cat([dark, middle, bright], dim=1)  # Join along the channel axis: [B, 3, H, W] x 3 becomes [B, 9, H, W].
        logits = self.network(inputs)  # Run the layers to produce [B, 3, H, W] exposure scores.
        weights = logits.softmax(dim=1)  # Convert scores into nonnegative blending weights that sum to 1 across the 3 exposures at each pixel.
        exposures = torch.stack([dark, middle, bright], dim=1)  # Add a separate exposure axis: [B, 3 exposures, 3 RGB channels, H, W].
        # unsqueeze(2) makes weights [B, 3 exposures, 1, H, W], sharing each exposure's weight across R, G, and B.
        # Multiplication weights each source pixel; summing over exposure axis 1 combines the three photos.
        return (exposures * weights.unsqueeze(2)).sum(dim=1)  # Return [B, 3, H, W]: a weighted RGB blend, not a newly generated scene.
