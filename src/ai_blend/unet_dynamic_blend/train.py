"""Supervised training: python -m ai_blend.unet_dynamic_blend.train --help."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from .dataset import ExposureDataset
from .loss import blend_loss
from .model import DynamicUNetBlend
from .utils import check_output, create_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cases"))
    parser.add_argument("--output", type=Path, default=Path("runs/unet_dynamic_blend"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.base_channels, args.threads) < 1:
        parser.error("epochs, base-channels and threads must be positive")
    if not 0 < args.scale <= 1 or not 0 < args.lr < float("inf"):
        parser.error("scale must be in (0, 1] and lr must be finite and positive")
    check_output(args.output, args.data)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    dataset = ExposureDataset(args.data, args.scale)
    for case in dataset.skipped_cases:
        print(f"Skipping missing target: {case}", flush=True)
    # Variable counts and resolutions: each batch contains one complete case.
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = DynamicUNetBlend(args.base_channels).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = create_run(args.output)
    history = []
    print(f"Training {len(dataset)} cases; saving to {run}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for index, (images, target) in enumerate(loader, start=1):
            started = perf_counter()
            images, target = images.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = blend_loss(model(images), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            value = loss.item()
            total += value
            print(
                f"Epoch {epoch + 1}/{args.epochs}, case {index}/{len(loader)} "
                f"| images={images.shape[1]} | loss={value:.6f} "
                f"| time={perf_counter() - started:.2f}s",
                flush=True,
            )
        history.append(total / len(loader))
        print(f"Epoch mean loss: {history[-1]:.6f}", flush=True)
        checkpoint = {
            "architecture": "dynamic_unet_blend",
            "base_channels": args.base_channels,
            "scale": args.scale,
            "epoch": epoch + 1,
            "lr": args.lr,
            "seed": args.seed,
            "weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_history": history,
        }
        # Replace only this run's checkpoint after a complete save.
        temporary = run / "model.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(run / "model.pt")
        (run / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Saved: {run / 'model.pt'}")


if __name__ == "__main__":
    main()
