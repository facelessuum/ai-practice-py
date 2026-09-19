"""Record software versions and source identities for a run."""
import platform
import subprocess
from pathlib import Path

import torch

from ..utils.save_files import file_hash


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
