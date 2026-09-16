"""Blend all cases into a new numbered output/runN directory."""

import argparse
from pathlib import Path

import torch

from .predict import blend_case, load_model


def create_run(root):
    """Allocate a fresh run directory without reusing an existing run."""
    root.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name[3:])
        for path in root.iterdir()
        if path.name.startswith("run") and path.name[3:].isascii() and path.name[3:].isdigit()
    ]
    number = max(numbers, default=0) + 1
    while True:
        run = root / f"run{number}"
        try:
            run.mkdir()
        except FileExistsError:
            number += 1
        else:
            return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cases"))
    parser.add_argument("--model", type=Path, default=Path("runs/ai_guide/unet.pt"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error(f"Cases directory does not exist: {args.data}")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}. Run 'uv run train-blend' first.")
    if args.output.resolve().is_relative_to(args.data.resolve()) or args.output.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Save predictions outside the input data")
    cases = sorted(path for path in args.data.iterdir() if path.is_dir())
    if not cases:
        parser.error(f"No cases found in {args.data}")

    torch.set_num_threads(4)
    model = load_model(args.model)
    run = create_run(args.output)
    print(f"Blending {len(cases)} cases into {run}", flush=True)
    for index, case in enumerate(cases, start=1):
        destination = run / case.name / "blend.png"
        blend_case(model, case, destination)
        print(f"[{index}/{len(cases)}] Saved: {destination}", flush=True)
    print(f"Finished: {run}")


if __name__ == "__main__":
    main()
