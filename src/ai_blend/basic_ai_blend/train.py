from time import perf_counter
from .model import ArNet
from torch.utils.data import DataLoader
from .dataset import ExposureDataset
import torch
from pathlib import Path
import argparse
from .loss import exposure_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cases"))
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")

    torch.manual_seed(69)
    torch.set_num_threads(10)
    dataset = ExposureDataset(args.data)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    base_channels = 8

    model = ArNet(base_channels=base_channels)
    print(f"ArNet:{sum(p.numel() for p in model.parameters()):,} trainable parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(
        f"Training {len(dataset)} cases at 15% scale on ryzen 5 5600g APU"
        + f"{torch.get_num_threads()} threads), {args.epochs} epochs.",
        flush=True,
    )

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        print(f"Epoch {epoch + 1}/{args.epochs}: Loading Cases...", flush=True)

        batches = iter(loader)

        for index, (dark, middle, bright, target) in enumerate(batches, start=1):
            started = perf_counter()

            label = f"Epoch {epoch + 1}/{args.epochs}, case {index}/{len(dataset)}"

            optimizer.zero_grad()
            output = model(dark, middle, bright)
            loss = exposure_loss(output, target)
            #            print(f"{label}: backward pass...", flush=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(dark)
            print(
                f"{label} | Time {perf_counter() - started:.1f}s | Loss={total / len(dataset):.6f} | Size={target.shape[-1]}x{target.shape[-2]}",
                flush=True,
            )

    output_dir = get_model_output_fold()
    checkpoint = {
        "architecture": "arnet",
        "base_channels": base_channels,
        "weights": model.state_dict(),
        "size": None,
    }
    torch.save(checkpoint, output_dir)

    print(f"Saved: {output_dir}")


def get_model_output_fold() -> Path:
    run_number = 1
    model_path = Path("model")
    while True:
        model_output = model_path / f"run_{run_number:04d}"
        try:
            model_output.mkdir()
        except FileExistsError:
            run_number += 1
            continue
        return model_output / "arnet.pt"
