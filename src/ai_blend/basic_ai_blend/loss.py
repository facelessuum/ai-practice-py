"""Lossss"""
from torch.fft import Tensor

import torch


def exposure_loss(output: Tensor, target: Tensor):
    return (output - target).square().mean()
