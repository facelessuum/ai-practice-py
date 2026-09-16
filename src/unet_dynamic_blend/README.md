# Dynamic U-Net blend pipeline

A supervised exposure-fusion pipeline supporting any positive number of aligned
RGB inputs. Shared U-Net layers and set-mean context predict per-pixel blending
weights. Reordering inputs does not change the result beyond rounding error.

## Files

- `model.py`: architecture, `model(images)` and `model.blend(*images)`.
- `dataset.py`: variable-count input loading and supervised targets.
- `loss.py`: target-based mean squared error.
- `train.py`: training, numbered checkpoints, and loss history.
- `predict.py`: single-case and batch prediction interface.
- `utils.py`: numbered runs, latest-checkpoint lookup, input-path safeguards.

No test files or changes to `src/app/` are included.

## Dataset layout

```text
data/cases/<case>/
  10_converted/     # one or more aligned RGB images
  50_ai_blend/
    output.jpg     # required for training only
```

Inputs must have matching dimensions and already be spatially aligned. Loading
`10_converted` does not perform alignment. Images are normalized to [0, 1] and
resized to 10% by default. Targets are resized to match the inputs; this does not
correct different framing or geometric misalignment. Training reports and skips
cases with missing targets; other malformed cases stop processing.

## Train

From the repository root:

```bash
PYTHONPATH=src uv run python -m unet_dynamic_blend.train --epochs 20
```

Options include `--data`, `--output`, `--base-channels`, `--scale`, `--lr`,
`--threads`, `--seed`, and `--device` (default `cpu`).

Each invocation starts fresh and creates `runs/unet_dynamic_blend/run_XXXX/`:

- `model.pt`: latest completed epoch's weights, optimizer state, settings,
  and `loss_history`.
- `loss_history.json`: per-epoch mean training MSE, one value per epoch.

Checkpoint saving happens after each epoch. Previous runs are not overwritten.
Resume training and validation/test splits are not implemented. Training loss
alone does not demonstrate generalization; compare models on held-out scenes.
Batch size is 1 because image count and resolution can differ between cases.

## Predict / interface

Single case (saves `output/unet_dynamic_blend/run_XXXX/ai_blend.jpg`):

```bash
PYTHONPATH=src uv run python -m unet_dynamic_blend.predict \
  --case data/cases/YOUR_CASE
```

All cases (saves `run_XXXX/<case>/ai_blend.jpg`):

```bash
PYTHONPATH=src uv run python -m unet_dynamic_blend.predict --data data/cases
```

Defaults to the highest numbered completed checkpoint under
`runs/unet_dynamic_blend`. Use `--model PATH` or `--model-root DIRECTORY` to
choose another. Other options: `--output`, `--scale`, `--device`, `--threads`.
Predictions use the training scale unless overridden; `--scale 1` uses original
input resolution and can require considerably more memory. Targets are never
loaded during prediction. A malformed case stops the command; prior results stay.

## Model API

```python
import torch
from unet_dynamic_blend import DynamicUNetBlend

model = DynamicUNetBlend(base_channels=16)
images = torch.rand(1, 5, 3, 65, 97)  # batch, count, RGB, height, width
output = model(images)               # [1, 3, 65, 97]
output = model.blend(*images.unbind(dim=1))
```

Within a tensor batch, all samples must have the same count and dimensions.
Images must share device and dtype; spatial dimensions must be at least 4 x 4.

A single input is returned unchanged and provides no learning signal for blending
weights. This model cannot invent detail, replace skies, or align images: it only
mixes source pixels. More images require more memory.

Fresh training is required; old ArNet/U-Net checkpoints are incompatible. Train
on varied input counts and evaluate on held-out scenes. A dataset containing only
three images per case still trains only on three-image examples, even though the
architecture accepts other counts. No automatic subset sampling is applied.

The package is used via `PYTHONPATH=src`; installed console commands and the root
packaging configuration have not been changed.
