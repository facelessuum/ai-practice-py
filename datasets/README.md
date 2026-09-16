# Scene-grouped dataset split

These directories contain relative symlinks to the original cases in
`data/cases/`. No source data was moved or modified.

- `train/`: 37 cases.
- `validation/`: 4 cases (all adobeDngSmall3mb and arwJob1 variants).
- `test/`: 3 cases (all largeMovement variants).

Groups were inferred from case-name prefixes, not verified scene identities.
The seven held-out cases have readable input JPEGs and target images.
This is a small, illustrative split: validation contains only two inferred
scene groups and test only one, so results will not establish broad quality.
Inspect scene identities before trusting leakage-free generalization metrics.

## Usage

Train a fresh U-Net++ run using ONLY the training split:

```bash
PYTHONPATH=src uv run python -m unet_plus_blend.train --data splits/train
```

Predict the validation split:

```bash
PYTHONPATH=src uv run python -m unet_plus_blend.predict --data splits/validation
```

After model choices are finalized, predict the test split:

```bash
PYTHONPATH=src uv run python -m unet_plus_blend.predict --data splits/test
```

Prediction commands save images but do not calculate evaluation metrics.
The current trainer does not automatically evaluate validation loss.
Never use validation/test directories as training data. The default trainer
path remains `data/cases`, so explicitly pass `--data splits/train`.
Checkpoints previously trained on all cases have already seen this holdout;
train from scratch on the training split for a meaningful evaluation.

Treat symlinked inputs as read-only. Keep generated predictions and checkpoints
outside both `data/` and these split directories.
