"""Train the U-Net blend model against the AI blend target."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from .dataset import ExposureDataset
from .loss import exposure_loss
from .unet import UNetBlendModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cases"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("runs/ai_guide/unet.pt"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.output.resolve().is_relative_to(
        args.data.resolve()
    ) or args.output.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Save the model outside data/")

    torch.manual_seed(42)
    torch.set_num_threads(10)
    dataset = ExposureDataset(args.data)
    # Cases have different dimensions: keep each full-resolution case separate.
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    base_channels = 16
    model = UNetBlendModel(base_channels=base_channels)
    print(f"U-Net: {sum(p.numel() for p in model.parameters()):,} trainable parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(
        f"Training {len(dataset)} cases at original resolution on CPU "
        + f"({torch.get_num_threads()} threads), {args.epochs} epochs.",
        flush=True,
    )
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        print(f"Epoch {epoch + 1}/{args.epochs}: loading cases...", flush=True)
        # Default collation stacks each of the sample's three tensors.
        batches = iter(loader)
        for index, (dark, middle, bright, target) in enumerate(batches, start=1):
            started = perf_counter()
            label = f"Epoch {epoch + 1}/{args.epochs}, case {index}/{len(dataset)}"
            print(
                f"{label}: {dark.shape[-1]}x{dark.shape[-2]} — forward pass...",
                flush=True,
            )
            optimizer.zero_grad()
            output = model(dark, middle, bright)
            loss = exposure_loss(output, target)
            print(f"{label}: backward pass...", flush=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(dark)
            print(
                f"{label}: loss={loss.item():.6f}, compute time={perf_counter() - started:.1f}s",
                flush=True,
            )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={total / len(dataset):.6f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "unet",
        "base_channels": base_channels,
        "weights": model.state_dict(),
        "size": None,
    }
    torch.save(checkpoint, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
