"""Train dynamic ArNet on a single v5e TPU using PyTorch/XLA."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .dataset import ExposureDataset
from .loss import blend_loss
from .model import ArNetDynamicModel
from .utils import check_output, create_run


class FixedSizeDataset(Dataset):
    """Resize on CPU before transfer to avoid spatial-shape recompilation."""

    def __init__(self, source, size):
        self.source = source
        self.size = (size, size)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        images, target = self.source[index]
        images = F.interpolate(
            images, size=self.size, mode="bilinear", align_corners=False, antialias=True
        )
        target = F.interpolate(
            target.unsqueeze(0), size=self.size, mode="bilinear",
            align_corners=False, antialias=True,
        )[0]
        return images, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/train"))
    parser.add_argument("--output", type=Path, default=Path("model/arnet_blend_tpu"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--size", type=int, default=None, help="Optional fixed square resolution; default preserves scaled dimensions")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    if min(args.epochs, args.base_channels) < 1 or (args.size is not None and args.size < 4):
        parser.error("epochs/channels must be positive and size must be >= 4")
    if not 0 < args.scale <= 1 or not 0 < args.lr < float("inf"):
        parser.error("scale must be in (0, 1] and lr finite and positive")
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
    source = ExposureDataset(args.data, args.scale)
    for case in source.skipped_cases:
        print(f"Skipping missing target: {case}", flush=True)
    dataset = FixedSizeDataset(source, args.size) if args.size is not None else source
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = ArNetDynamicModel(args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False)
    run = create_run(args.output)
    history = []
    print(f"Device: {device} ({xr.device_type()}); cases: {len(source)}; output: {run}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("The first steps compile XLA graphs and may be slow.", flush=True)
    if args.size is None:
        print("Using scaled image dimensions; different shapes can trigger TPU recompilation.", flush=True)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for index, (images, target) in enumerate(loader, start=1):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = blend_loss(model(images), target)
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
            "scale": args.scale,
            "training_size": args.size,
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
