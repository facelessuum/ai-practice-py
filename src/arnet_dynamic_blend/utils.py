from pathlib import Path

"""Numbered runs and safeguards for read-only input data."""


def check_output(path: Path, input_root: Path) -> None:
    resolved = path.resolve()
    for protected in (Path("data").resolve(), input_root.resolve()):
        if resolved.is_relative_to(protected):
            raise ValueError(f"Save generated files outside input data: {path}")


def create_run(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    numbers = [
        int(path.name.removeprefix("run_"))
        for path in root.glob("run_*")
        if path.is_dir() and path.name.removeprefix("run_").isdigit()
    ]

    number = max(numbers, default=0) + 1

    while True:
        run = root / f"run_{number:04d}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            number += 1


def latest_checkpoint(root: Path) -> Path:
    paths = [
        path
        for path in root.glob("run_*/arnet_blend.pt")
        if path.is_file() and path.parent.name.removeprefix("run_").isdigit()
    ]
    if not paths:
        raise FileNotFoundError(f"No run_*/arnet_blend.pt checkpoints in {root}")

    return max(paths, key=lambda path: int(path.parent.name.removeprefix("run_")))
