"""Never reuse a previous run folder."""
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .record_run_details import environment
from ..utils.save_files import write_json

DEFAULT_OUTPUT = "output/window_frame/ai_guide"


def create_run(kind, output_root=DEFAULT_OUTPUT):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root).resolve() / kind / f"{stamp}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "metadata.json", dict(kind=kind, created_utc=stamp, environment=environment()))
    logger = logging.getLogger(str(folder))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.FileHandler(folder / "run.log"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.info("Run folder: %s", folder)
    return folder, logger
