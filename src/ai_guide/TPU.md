# Single-device v5e-1 TPU training

Branch: `feat/arnet-tpu-v5e`. Uses PyTorch/XLA, not CUDA. The ordinary ArNet
trainer remains unchanged; use the dedicated `train-arnet-tpu` command.

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
   PJRT_DEVICE=TPU uv run train-arnet-tpu --epochs 20 --size 256
   ```

   Or run directly from source:

   ```bash
   PYTHONPATH=src PJRT_DEVICE=TPU uv run python -m ai_guide.train_tpu --epochs 20
   ```

## Behavior and limitations

- Single XLA device only. Multi-chip/multi-host TPU training is not implemented.
- The original ArNet architecture and supervised MSE loss are reused.
- Inputs are loaded at `--scale` then inputs and targets are resized on CPU to
  `--size` square. This distorts aspect ratios but stabilizes spatial shapes for
  compilation. The transform is applied identically to inputs and targets.
- Batch size is one case. Different exposure counts still create different graph
  shapes and can trigger recompilation. The first steps compile and may be slow.
- Uses `xm.optimizer_step(..., barrier=True)` for lazy graph execution and
  `xm.save` for CPU-portable checkpoints. Per-case loss logging synchronizes with
  the TPU; this is a simple baseline, not a throughput-optimized trainer.
- Each run saves under `model/arnet_blend_tpu/run_XXXX/`, including epoch mean
  `loss_history.json` and `arnet_blend.pt`. Each completed epoch replaces that
  run's checkpoint. There is no resume command or early stopping yet.
- Checkpoints record `training_size`. Existing inference code does not use that
  field automatically. To match training preprocessing, resize inputs to the
  same square size. The fully convolutional model can accept other dimensions,
  but assess quality when changing resolution/aspect ratio.
- Colab local storage is temporary. Use `--output` with a persistent mounted
  directory to preserve completed checkpoints if the runtime is reclaimed.
- TPU execution must be verified on the actual v5e-1 runtime; local syntax
  checks are not a substitute for a hardware run.
