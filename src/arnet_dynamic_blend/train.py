"""Train dynamic ArNet on a single v5e TPU using PyTorch/XLA."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import CropExposureDataset
from .loss import blend_loss
from .model import ArNetDynamicModel
from .utils import check_output, create_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/train"))
    parser.add_argument("--output", type=Path, default=Path("model/arnet_blend_tpu"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    if min(args.epochs, args.base_channels) < 1 or min(args.crop_width, args.crop_height) < 4:
        parser.error("epochs/channels must be positive and crop dimensions must be >= 4")
    if not 0 < args.lr < float("inf"):
        parser.error("lr must be finite and positive")
    check_output(args.output, args.data)

    # Explicit TPU backend: do not silently fall back to CPU or CUDA.
    os.environ["PJRT_DEVICE"] = "TPU"
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "9"]:
        raise RuntimeError("This branch requires matching torch 2.9 / torch-xla 2.9")
    device = torch_xla.device()
    if xr.device_type() != "TPU":
        raise RuntimeError("Select a TPU runtime before running this command")
    if xr.global_runtime_device_count() != 1:
        raise RuntimeError("This trainer supports a single-device v5e-1 runtime only")
    torch.manual_seed(args.seed)
    xm.set_rng_state(args.seed, device)
    source = CropExposureDataset(args.data, args.crop_width, args.crop_height)
    for case in source.skipped_cases:
        print(f"Skipping missing target: {case}", flush=True)
    loader = DataLoader(source, batch_size=1, shuffle=True)
    model = ArNetDynamicModel(args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False)
    run = create_run(args.output)
    history = []
    print(f"Device: {device} ({xr.device_type()}); cases: {len(source)}; output: {run}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("The first steps compile XLA graphs and may be slow.", flush=True)
    print(f"Original-resolution crops: {args.crop_width}x{args.crop_height}; padded pixels masked.", flush=True)
    print("Different exposure counts can still trigger TPU recompilation.", flush=True)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for index, (images, target, mask) in enumerate(loader, start=1):
            images, target, mask = images.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = blend_loss(model(images), target, mask)
            loss.backward()
            # barrier executes the lazy graph; plain optimizer.step is not enough.
            xm.optimizer_step(optimizer, barrier=True)
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError("Non-finite loss; current epoch checkpoint was not saved")
            total += value
            print(f"Epoch {epoch + 1}/{args.epochs}, case {index}/{len(loader)} | loss={value:.6f}", flush=True)
        history.append(total / len(loader))
        checkpoint = {
            "architecture": "arnet_blend",
            "base_channels": args.base_channels,
            "scale": 1.0,
            "preprocessing": "random_crop_with_edge_padding",
            "crop_width": args.crop_width,
            "crop_height": args.crop_height,
            "epoch": epoch + 1,
            "lr": args.lr,
            "seed": args.seed,
            "weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_history": history,
        }
        temporary = run / "arnet_blend.tmp"
        # XLA-aware save transfers tensors to CPU for portable checkpoints.
        xm.save(checkpoint, temporary)
        temporary.replace(run / "arnet_blend.pt")
        (run / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"Epoch mean: {history[-1]:.6f}; checkpoint saved", flush=True)


if __name__ == "__main__":
    main()
