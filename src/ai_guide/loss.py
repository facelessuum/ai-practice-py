"""Supervised reconstruction loss against the AI blend target."""

import torch


def exposure_loss(output, target):
    """Mean-squared error between the prediction and the target blend."""
    return (output - target).square().mean()
