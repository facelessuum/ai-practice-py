"""Image-only, two-decoder network. No pretrained weights or external models."""
import torch
from torch import nn
from torch.nn import functional as F
from .model_layers import ConvBlock, Decoder

CLASS_NAMES = ["background", "clean_neutral", "neutral", "wood", "dark", "protected"]


def compose(image, residual, class_probabilities, frame_probability, confidence=0.8):
    """Copy excluded pixels exactly. Protection depends on prediction accuracy."""
    certainty, labels = class_probabilities.max(dim=1, keepdim=True)
    allowed = (labels >= 1) & (labels <= 4) & (certainty >= confidence)
    allowed &= frame_probability >= confidence
    gate = allowed.to(image.dtype) * frame_probability
    enhanced = torch.where(allowed, (image + gate * residual).clamp(0, 1), image)
    return enhanced, gate


class WindowFrameModel(nn.Module):
    def __init__(self, base_channels=32, max_correction=0.25, confidence=0.8):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")
        self.settings = dict(base_channels=base_channels, max_correction=max_correction,
                             confidence=confidence)
        widths = [base_channels * 2**i for i in range(4)]
        self.encoder = nn.ModuleList([ConvBlock(a, b) for a, b in zip([3] + widths[:-1], widths)])
        self.segmentation = Decoder(widths)
        self.enhancement = Decoder(widths)
        self.class_head = nn.Conv2d(widths[0], 6, 1)
        # Separate head: frame supervision remains useful when material is unknown.
        self.mask_head = nn.Conv2d(widths[0], 1, 1)
        self.correction_head = nn.Sequential(ConvBlock(widths[0] + 7, widths[0]),
                                             nn.Conv2d(widths[0], 3, 1))
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(self, image):
        features = []
        x = image
        for i, block in enumerate(self.encoder):
            x = block(F.avg_pool2d(x, 2) if i else x)
            features.append(x)
        segmentation = self.segmentation(features)
        class_logits = self.class_head(segmentation)
        mask_logits = self.mask_head(segmentation)
        probabilities = class_logits.softmax(1)
        soft_mask = mask_logits.sigmoid()
        # Detach conditioning so image losses cannot redefine the class meanings.
        conditioned = torch.cat((self.enhancement(features), probabilities.detach(),
                                 soft_mask.detach()), dim=1)
        residual = self.correction_head(conditioned).tanh() * self.settings["max_correction"]
        enhanced, edit_gate = compose(image, residual, probabilities, soft_mask,
                                      self.settings["confidence"])
        return dict(class_logits=class_logits, mask_logits=mask_logits,
                    probabilities=probabilities, soft_mask=soft_mask, residual=residual,
                    enhanced=enhanced, edit_gate=edit_gate)
