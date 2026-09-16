# Basic AI blend

Three aligned RGB photos in → one blended RGB photo out.

## Files

- `unet.py`: two-level U-Net that predicts blending weights; the default training model.
- `model.py`: original tiny two-layer CNN, retained for loading older checkpoints.
- `dataset.py`: reads only the three JPEGs from each `30_aligned/` folder.
- `loss.py`: supervised mean-squared reconstruction loss against `50_ai_blend/output.jpg`.
- `train.py`: CPU training and checkpoint saving.
- `predict.py`: generates a blend with saved weights.
- `blend_all.py`: blends every case into a new numbered output run.
- Commands are registered in the root `pyproject.toml`.

## Train (when you are ready)

```bash
uv run train-blend --epochs 20
```

Uses the existing virtual environment, PyTorch, NumPy, and Pillow. Saves `runs/ai_guide/unet.pt`. Each run starts fresh and overwrites that checkpoint, leaving the old `model.pt` untouched. U-Net needs fresh training; tiny-CNN weights cannot initialize it.

Warning: training still uses full-resolution images on CPU. U-Net uses much more memory and compute than the tiny CNN; large photos may exhaust RAM. Patch training is not implemented here.

## Blend all cases

After training, run from the repository root:

```bash
uv run uv-blend
```

Processes every case directory in `data/cases/` (not limited to 20). Uses the saved model without training. Results retain the original resolution of the `30_aligned/` images, without resizing or tiling:

```text
output/
  run1/
    <case-name>/blend.png
    <another-case-name>/blend.png
  run2/
    ...
```

Each invocation creates the next run number without overwriting previous runs. Optional flags: `--data`, `--model`, `--output` (the output root). A malformed case stops the command; any completed results remain in that run.

## Predict one case

From the repository root:

```bash
uv run predict-blend \
  --case data/cases/jonaEasyTest_g23_5060ti_zed
```

Loads `runs/ai_guide/unet.pt` by default and saves `runs/ai_guide/blend.png` at the original aligned input resolution. Both prediction commands also accept `--model runs/ai_guide/model.pt` to use an existing tiny-CNN checkpoint without retraining. Checkpoint metadata selects the appropriate architecture.

## Supervised training

Each case loads `50_ai_blend/output.jpg` as the target. The model compares its prediction with that target using mean-squared error, so this is supervised learning. The model learns to imitate the existing AI blend outputs; it does not necessarily learn a general professional editing style.

Inputs must already be aligned. Numbered files retain the exported bracket order. Training and prediction both process the full aligned input resolution without resizing, cropping, or tiling. Training loads one case at a time with batch size 1 because cases have different dimensions. Full-resolution training can use substantially more memory than prediction. This baseline does not perform HDR radiance reconstruction, colour correction, or sky replacement, and has no validation setup or production-quality guarantees.

`src/app/` is user-owned and untouched. Root `AGENTS.md` protects it.
