"""Check training settings and checkpoint compatibility."""
from dataclasses import asdict
import json
from pathlib import Path

from ..training.configure_hardware import resource_plan
from ..training.check_early_stopping import EarlyStopping


def validate_config(config):
    data, training, model = config.dataset, config.training, config.model
    for key in ("epochs", "batch_size", "accumulation_steps"):
        if getattr(training, key) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ('max_train_samples', 'max_validation_samples'):
        limit = getattr(training, key)
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError(f'{key} must be a positive integer')
    if data.image_size < 16:
        raise ValueError("image_size must be >=16")
    if not isinstance(training.skip_audit, bool):
        raise ValueError("skip_audit must be a boolean")
    resource_plan(training, "cpu")
    EarlyStopping(training.early_stopping_patience, training.early_stopping_min_delta)
    if not 0 < model.confidence <= 1 or not 0 < model.max_correction <= 1:
        raise ValueError("confidence and max_correction must be in (0, 1]")
    if min(training.segmentation_epochs, training.enhancement_epochs) < 0:
        raise ValueError("Stage lengths cannot be negative")


def check_resume(config, state, checkpoint, samples):
    """A resumed experiment must use the same data and model."""
    saved = state["config"]
    if saved["dataset"] != asdict(config.dataset) or saved["model"] != asdict(config.model):
        raise ValueError("Resume requires the same dataset and model settings")
    for name in ("segmentation_epochs", "enhancement_epochs", "batch_size", "accumulation_steps", "seed",
                 "max_train_samples", "max_validation_samples"):
        if saved["training"].get(name) != getattr(config.training, name):
            raise ValueError(f"Resume requires the same {name}; start a new run to change it")
    audit_path = Path(checkpoint).resolve().parent.parent / "dataset_audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text())["samples"] != samples:
        raise ValueError("Cannot verify unchanged dataset against the original audit")
