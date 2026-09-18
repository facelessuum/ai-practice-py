"""Versioned checkpoints, atomic saves, and sidecar metadata."""
import random
from pathlib import Path
import numpy as np
import torch
from ..model.window_frame_model import WindowFrameModel, CLASS_NAMES
from ..run_history.record_run_details import write_json, file_hash, environment


def save_checkpoint(path, model, optimizer, epoch, best_score, config, split, metrics, generator,
                    early_stopping=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               loader=generator.get_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    provenance = environment()
    state = dict(format_version=1, environment=provenance, model=model.state_dict(), model_settings=model.settings,
                 optimizer=optimizer.state_dict(), epoch=epoch, best_score=best_score,
                 config=config, split=split, metrics=metrics, class_names=CLASS_NAMES, rng=rng,
                 early_stopping=early_stopping)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), dict(format_version=1, epoch=epoch, best_score=best_score,
        sha256=file_hash(path), model_settings=model.settings, config=config, metrics=metrics,
        class_names=CLASS_NAMES, environment=provenance, early_stopping=early_stopping,
        training_run=str(path.parent.parent)))


def load_checkpoint(path, device="cpu"):
    # Includes optimizer/RNG Python objects. Only open checkpoints you trust.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = WindowFrameModel(**state["model_settings"])
    model.load_state_dict(state["model"])
    return model.to(device), state


def restore_random_state(state, generator):
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    generator.set_state(rng["loader"])
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
