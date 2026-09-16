"""Supervised training for the dynamic ArNet blend model."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from .dataset import ExposureDataset
from .loss import blend_loss
from .model import ArNetDynamicModel
from .utils import check_output, create_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/train"))
    parser.add_argument("--output", type=Path, default=Path("model/arnet_blend"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    if min(args.epochs, args.base_channels) < 1:
        parser.error("epochs and base-channels must be positive")
    if not 0 < args.scale <= 1 or not 0 < args.lr < float("inf"):
        parser.error("scale must be in (0, 1] and lr must be finite and positive")

    check_output(args.output, args.data)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ExposureDataset(args.data, args.scale)

    for case in datasets.skipped_cases:
        print(f"Skipping missing target: {case}", flush=True)

    loader = DataLoader(datasets, batch_size=1, shuffle=True)
    model = ArNetDynamicModel(args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = create_run(args.output)

    history = []
    print(f"Training {len(datasets)} cases; saving to {run}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()

    for epoch in range(args.epochs):
        total = 0.0
        for index, (images, target) in enumerate(loader, start=1):
            started = perf_counter()
            images, target = images.to(device), target.to(device)
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
            "architecture": "arnet_blend",
            "base_channels": args.base_channels,
            "scale": args.scale,
            "lr": args.lr,
            "seed": args.seed,
            "weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_history": history,
        }
        temporary = run / "arnet_blend.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(run / "arnet_blend.pt")
        (run / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Saved: {run / 'arnet_blend.pt'}")



if __name__ == "__main__":
    main()
