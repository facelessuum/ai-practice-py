"""Device selection and random seeds shared across the pipeline."""
import random
import warnings

import numpy as np
import torch


def choose_device(requested="auto"):
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        return device
    # Reuse the project's global device. Its current 'cude' typo fails on CUDA hosts.
    try:
        from utils.utils import DEVICE
        return DEVICE
    except (ImportError, RuntimeError, ValueError) as error:
        warnings.warn(f"Shared DEVICE unavailable ({error}); using local safe selection.")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
