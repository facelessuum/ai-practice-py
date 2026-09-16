# Single-device v5e-1 TPU training

Branch: `feat/arnet-tpu-v5e`. Uses PyTorch/XLA, not CUDA. On this branch,
`src/arnet_dynamic_blend/train.py` is the TPU trainer. Run `train-arnet-blend`;
`train-arnet-tpu` remains a compatibility alias. Other models are unchanged.
The trainer selects the TPU backend automatically; there is no CPU/CUDA fallback.

## Colab terminal

1. Select a **v5e-1 TPU** runtime. Changing runtimes may replace the VM; preserve
   code/data/checkpoints in persistent storage first.
2. Fetch this branch after it has been pushed, and enter the repository:

   ```bash
   git fetch origin
   git switch feat/arnet-tpu-v5e
   uv sync
   ```

   PyTorch and torch-xla are pinned to 2.9.0 together with the TPU runtime extra.
   Do not independently upgrade one. No CPU-only wheel index is used. Regular
   PyTorch wheels may install CUDA dependencies, but this command uses XLA/TPU.
   This environment targets a Linux TPU VM, not a local machine without TPU access.

3. Check the device:

   ```bash
   PJRT_DEVICE=TPU uv run python -c 'import torch_xla; import torch_xla.runtime as xr; print(torch_xla.device(), xr.device_type())'
   ```

   Expect an XLA device and `TPU`. `torch.cuda.is_available()` is not a TPU check.

4. Train (ensure `datasets/train` symlinks and source images exist on the VM):

   ```bash
   uv run train-arnet-blend --epochs 20
   ```

   Or run directly from source:

   ```bash
   PYTHONPATH=src uv run python -m arnet_dynamic_blend.train --epochs 20
   ```

## Behavior and limitations

- Single XLA device only. Multi-chip/multi-host TPU training is not implemented.
- Defaults: target-resolution random crops, **512 wide x 512 high**, and
  32 base channels. `--scale` and `--size` have been replaced by `--crop-width`
  and `--crop-height`.
- Inputs are resized with LANCZOS to the unchanged target's dimensions if needed.
  Aspect ratios must match within 0.1% (rounding tolerance); otherwise loading
  fails rather than stretching the scene. All input exposures must have the same
  source dimensions. Downsampling loses some input detail; upsampling does not
  create new detail. Targets are never resized by this crop loader.
- Each case samples one new location per epoch. Exactly the same crop coordinates
  are used for every normalized exposure and its target. Images must already
  have matching framing/alignment: matching aspect ratios do not guarantee this.
  Crops retain target-resolution detail but lose whole-scene context.
- Images smaller than the crop are edge-padded on the bottom/right. A validity
  mask excludes padding from MSE loss; RGB errors are averaged over real pixels.
- Batch size is one case. Height and width are now fixed, but different exposure
  counts still change graph shapes and can trigger recompilation. The first steps
  compile and may be slow. Fixed-size crops do NOT guarantee fitting TPU memory. If
  needed use `--crop-width 256 --crop-height 256` (no downsampling).
- Uses `xm.optimizer_step(..., barrier=True)` for lazy graph execution and
  `xm.save` for CPU-portable checkpoints. Per-case loss logging synchronizes with
  the TPU; this is a simple baseline, not a throughput-optimized trainer.
- Each run saves under `model/arnet_blend_tpu/run_XXXX/`, including epoch mean
  `loss_history.json` and `arnet_blend.pt`. Each completed epoch replaces that
  run's checkpoint. There is no resume command or early stopping yet.
- Checkpoints record crop dimensions, preprocessing type, and `scale=1.0`.
  This scale means no global fractional resize; training still normalizes input
  resolution to the target. Target-free inference must choose its working
  resolution explicitly and assess quality at that resolution.
  Inference tiling is not implemented here. The fully convolutional model can
  accept larger images, but full-resolution prediction may exceed memory;
  overlapping-tile inference would need to be added separately. Evaluate on
  deterministic held-out crops or complete scenes rather than random training
  crops when comparing model quality.
- Colab local storage is temporary. Use `--output` with a persistent mounted
  directory to preserve completed checkpoints if the runtime is reclaimed.
- TPU execution must be verified on the actual v5e-1 runtime; local syntax
  checks are not a substitute for a hardware run.
