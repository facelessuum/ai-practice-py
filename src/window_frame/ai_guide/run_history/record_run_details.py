"""JSON records, file identities, reproducibility, and safe device selection."""
import hashlib
import json
import platform
import random
import subprocess
import warnings
from pathlib import Path
import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    source = Path(__file__).resolve().parents[1]
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                cuda=torch.version.cuda, git_commit=commit, git_dirty=dirty,
                source_hashes={str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*.py"))})


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
