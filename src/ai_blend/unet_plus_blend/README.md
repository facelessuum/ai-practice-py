# U-Net++ blend model

Variable-count RGB exposure fusion using shared U-Net++ features. Unlike the
ordinary U-Net, decoder nodes combine **all previous nodes at their resolution**
with upsampled features from the next level. Default depth is three downsampling
steps (four resolutions). This is a compact U-Net++ variant without deep
supervision, a pretrained encoder, or attention.

## Usage

Run Python from the repository root with `PYTHONPATH=src`:

```python
import torch
from ai_blend.unet_plus_blend import UNetPlusBlend
from ai_blend.unet_plus_blend.loss import blend_loss

model = UNetPlusBlend(base_channels=16, depth=3)
images = torch.rand(1, 5, 3, 65, 97)  # batch, count, RGB, height, width
output = model(images)               # [1, 3, 65, 97]
output = model.blend(*images.unbind(dim=1))

# Replace these random inputs/target with real aligned training data.
target = torch.rand(1, 3, 65, 97)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss = blend_loss(output, target)
loss.backward()
optimizer.step()
```

## Pipeline

- `dataset.py`: loads variable-count exposures from `<case>/10_converted/` and
  supervised targets from `<case>/50_ai_blend/output.jpg`.
- `loss.py`: supervised mean squared error.
- `train.py`: training, numbered runs, checkpoints, and loss history.
- `predict.py`: single-case and batch prediction, without requiring targets.
- `utils.py`: run allocation, checkpoint lookup, and read-only input safeguards.

### Train

```bash
uv run python -m ai_blend.unet_plus_blend.train --epochs 20 --depth 3
```

Defaults: CPU, 16 base channels, learning rate 0.001, and 10% image scale.
Batch size is 1 to support varying counts and dimensions. Cases without targets
are reported and skipped; malformed inputs fail with an error. Inputs must
already be aligned. Targets are resized to match inputs, which does not correct
geometric misalignment.

Every invocation starts fresh in `runs/unet_plus_blend/run_XXXX/`. After each
epoch, `model.pt` saves weights, optimizer state, depth, channel count, scale,
and training loss history. `loss_history.json` stores epoch mean losses for
plotting. No resume CLI or validation split is implemented. Training loss alone
is not evidence of generalization. The pipeline uses the available exposure
counts; it does not randomly subsample exposures.

### Predict

```bash
# One case: output/unet_plus_blend/run_XXXX/ai_blend.jpg
uv run python -m ai_blend.unet_plus_blend.predict --case data/cases/YOUR_CASE

# All cases: output/unet_plus_blend/run_XXXX/<case>/ai_blend.jpg
uv run python -m ai_blend.unet_plus_blend.predict --data data/cases
```

Prediction defaults to the highest numbered completed U-Net++ checkpoint and
restores its channel count, depth, and scale. Use `--model PATH` to choose one.
`--scale 1` predicts at original input resolution and may need much more memory.
A malformed case stops batch prediction, preserving already saved outputs.
Both commands accept `--output`, `--device`, and `--threads`; use `--help` for
all options. Run with `PYTHONPATH=src`; installed console scripts are unchanged.

## Constraints

- Accepts any positive count; no image ordering is assumed. Set-mean context
  makes fusion permutation-invariant up to floating-point rounding.
- Inputs must already be aligned, normalized floating-point RGB, and have
  matching shapes, dtypes and devices. Minimum spatial size is `2**depth`.
  Odd dimensions are supported using explicit upsampling sizes.
- Each sample within a tensor batch must have the same count and dimensions.
- The output is a convex blend of the input pixels, not arbitrary generated RGB.
  It cannot replace skies, invent missing detail, or align inputs. One image is
  returned unchanged and provides no blending-weight learning signal.
- Nested skip paths generally cost more memory and compute than a comparable
  plain U-Net. U-Net++ does not guarantee better results; evaluate held-out scenes.
- Train from scratch on varied input counts. ArNet and ordinary U-Net checkpoints
  are incompatible. Architecture flexibility alone does not guarantee quality
  for input counts absent from training.

No test files are included, and existing models and pipelines are unchanged.
