# Your window-frame model

Give it **one photo**. It predicts frame locations, frame types, and small improvements to eligible frames. Everything is trained from scratch: no downloaded weights, SAM3, or external enhancement model.

This is a working training pipeline, **not an already-trained model**. Good boundaries and attractive enhancements must be demonstrated on photos it did not train on.

## Where things live

| Folder | Purpose |
| --- | --- |
| `settings/` | Settings you can change between experiments |
| `dataset/` | Read, check, and split examples |
| `model/` | Your model and its building blocks |
| `training/` | Teach the model, calculate mistakes, save progress |
| `evaluation/` | Measure results on held-out examples |
| `prediction/` | Enhance new photos and save outputs |
| `visualization/` | Colored guides, comparisons, and training charts |
| `run_history/` | Separate run folders, settings, logs, and file identities |
| `tests/` | Small automated checks using artificial examples |

The model has one shared image-reading section and two branches: one learns where/what the frames are; the other learns small RGB corrections. A separate mask head allows frame-location supervision when a material label is unknown. The default widths are 32, 64, 128, and 256. You don't need to change these to start.

## 1. Setup

Run commands from `/home/umm/projects/ai-practice-py` (the project root).

Use your project's existing environment. After installing/syncing the project, these commands are available:

```bash
uv sync
train-window-frame --help
evaluate-window-frame --help
predict-window-frame --help
```

If commands aren't on your shell's PATH, use `.venv/bin/train-window-frame`, etc. `uv sync` normally updates `uv.lock`; that is an installation action, not needed for the module-style commands below when dependencies are already installed.

Without reinstalling the project, use:

```bash
PYTHONPATH=src .venv/bin/python -m window_frame.ai_guide.training.train_model --help
```

For an NVIDIA RTX 50-series card, use a current PyTorch CUDA build supporting your GPU. CPU works for small tests but is slow for full training. Start at 512×512, batch size 4; reduce the batch size if you run out of GPU memory. Actual memory use depends on your settings.

The project's shared `utils.utils.DEVICE` is reused where possible. Its current `"cude"` typo would fail on a CUDA machine, so this package falls back safely without modifying the shared file. Setting `device="cuda"` selects CUDA explicitly.

## 2. Check the training examples

Each sample folder must contain:

```text
00001/
├── input.jpg          # input.png / input.jpeg also supported
├── output.png         # desired result
├── mask.png           # grayscale frame mask, 0–255
└── rail_classes.png   # raw IDs, not an RGB overlay
```

`rail_guide.png` is not a model input or training label. It is only a visual aid.

This implementation expects the following meanings:

| ID | Meaning | Editing |
| --- | --- | --- |
| 0 | Background | Copy input |
| 1 | Clean neutral | Learn minimal changes from target |
| 2 | Neutral | Learn changes from target |
| 3 | Wood | Learn changes while preserving wood appearance |
| 4 | Dark | Learn changes while retaining dark appearance |
| 5 | Colored/protected | Copy input |
| 255 | Unknown material | Ignore class supervision |

**Numeric IDs alone do not prove these meanings.** Confirm them against the original label-producing process. Class-specific appearance is learned from your targets, not guaranteed by the class names. If your current IDs have different meanings, do not train with this mapping.

A nonzero frame-mask pixel with rail ID 0 is treated as unknown material, rather than teaching the model that a frame is background. The separate frame mask still teaches its location. Original dataset files are never rewritten.

An empty mask should mean there is no frame, not that the frame was left unannotated. Class-5 target pixels should be identical to the input. Uncertain predictions are left unedited.

Check the dataset without training:

```bash
train-window-frame --audit-only
```

Or, without installed command shortcuts:

```bash
PYTHONPATH=src .venv/bin/python -m window_frame.ai_guide.training.train_model --audit-only
```

The audit records bad files, dimensions, class counts, empty masks, unknown frame pixels, protected target differences, and file hashes. Hashing the full dataset can take a minute or more.

Initial read-only audit of the current dataset found:

- 4,259 valid paired examples; no structural errors.
- 1,907 empty masks.
- Raw rail IDs 0–5.
- Three excess exact-duplicate input images.
- Existing class-5 pixels have zero input/target difference.
- Many nonzero mask pixels have rail ID 0; these need unknown-label handling.

This is not a manual assessment of annotation quality or confirmation of class names.

The current dataset mapping has been reviewed and confirmed by the user. The default settings already contain `class_meanings_confirmed = true`. Training does not ask any yes/no questions:

```bash
uv run train-window-frame
```

Older custom settings with `class_meanings_confirmed = false` still need that value updated, or the existing `--confirm-labels` flag. File and label-validity checks remain enabled.

## 3. Keep similar scenes together

By default, exact duplicate images stay in the same split. This **does not detect every near-duplicate or another view of the same property**.

For reliable evaluation, supply a JSON file mapping every sample ID to its property/scene:

```json
{"00001": "property_a", "00002": "property_a", "00003": "property_b"}
```

Set `dataset.scene_groups` to that file's path. All samples from a group stay together, and duplicate groups are joined. The saved split is reused for checkpoint evaluation and resuming. Without scene IDs the program logs a leakage warning. Fractions are approximate because groups stay intact.

The existing `datasets/window_frame/test/` contains photos without paired labels. Use those for **prediction**, not scored evaluation. Scored evaluation uses the labeled test portion held out from `train/`.

## 4. Train

```bash
train-window-frame --config src/window_frame/ai_guide/settings/training.toml
```

For a quick small-data test without moving photos:

```bash
uv run train-window-frame --epochs 5 --max-train-samples 50 --max-validation-samples 10
```

These limits select reproducible subsets after splitting; the held-out test split is unchanged. The chosen examples and effective settings are saved in the run. The initial audit still checks the full dataset. Limits are maximums, so a smaller split uses all its available examples. To resume a limited run, use the same sample-limit arguments and increase `--epochs` to the desired total (not additional) epoch count.

**Five epochs with the default stage settings only train detection/classification.** To exercise all stages in a five-epoch smoke test, use a separate config with `segmentation_epochs=1` and `enhancement_epochs=1`. A 50-example run is a pipeline check, not evidence of production quality.

The default schedule is:

1. Eight epochs learning frame locations and classes.
2. Eight epochs adding enhancement using known frame locations. Location/class learning continues.
3. Remaining epochs learning enhancement with predicted locations/classes.

An epoch is one pass through the training examples. Validation examples are used to check progress, not to update the model.

The unchanged background is not allowed to dominate the enhancement loss. Protected/background correction is penalized. Unknown material pixels do not contribute to material or enhancement target supervision. A soft predicted edit gate is used during joint learning; exported predictions use a stricter confidence gate, and evaluation measures those actual exported-style edits.

`best.pt` is selected by the same joint validation loss across all stages, not by test scores. Check enhancement metrics too: a model that makes no changes can look good on a mostly unchanged dataset. The best loss is not automatically the best-looking image.

Useful settings:

- `epochs`: total passes through the training data.
- `batch_size`: photos processed together; reduce if GPU memory runs out.
- `accumulation_steps`: combines several small batches before updating weights.
- `learning_rate`: how large a learning step is.
- `confidence`: certainty required before permitting an edit.
- `max_correction`: largest possible RGB correction, with colors scaled from 0 to 1.
- `base_channels`: model width; 8 for tiny tests, 32 as a practical starting point.
- `mixed_precision`: uses BF16 on supported CUDA hardware to reduce memory.

Save alternative settings under `settings/` so experiments are repeatable. The effective settings are also copied into every run.

### Automatic CPU/GPU performance settings

`cpu_threads`, `workers`, and `audit_workers` default to `"auto"`:

- On CPU, available logical CPUs are divided between model calculations and image-loading workers. Process affinity and the common Linux container quota are respected. A single CPU still works.
- Dataset checks run in parallel, with up to eight threads to avoid excessive memory/disk contention.
- On GPU, CUDA runs model calculations; CPU workers prepare batches ahead of time. Pinned-memory transfers, non-blocking copies, channels-last layout, supported BF16, cuDNN algorithm selection, and TF32 can improve throughput.
- Worker counts are capped rather than creating hundreds of image-loading processes. More workers/threads are not always faster; you can override each setting with a number. Use `workers=0` for troubleshooting multiprocessing.

Each training run records its selected settings in `hardware.json`, logs its device/thread counts, and records training images per second and peak GPU memory in `history.json`. `batch_size` remains explicit: increase it while monitoring memory and throughput. There is no unsafe "use every byte" switch or promise of 100% GPU utilization. Only the selected GPU is used; this is not distributed multi-GPU training.

GPU speed options can change numerical results slightly. Disable `allow_tf32` and `cudnn_benchmark` if you need stricter numerical comparisons. This alone does not make every operation deterministic. Tiny automated tests explicitly use one compute thread; production defaults do not.

### Total remaining time and early stopping

Progress now shows both the current-pass ETA and `all epochs ~...`: an estimate for the entire remaining planned run, including validation. The first estimate is rough (it assumes similar batch times); after a complete epoch it uses recent measured epoch durations, including checkpoint saving. Later training stages can take longer, so estimates will change. The estimate assumes all planned epochs run; it cannot predict when early stopping will trigger.

```toml
[training]
early_stopping_patience = 5
early_stopping_min_delta = 0.0001
```

These settings stop training after five consecutive monitored epochs without a validation-loss decrease greater than 0.0001. Set patience to `0` to disable early stopping. The first two preparatory stages always finish; monitoring starts fresh at the final joint stage. With the default 8+8 preparatory epochs, monitoring begins at epoch 17. If your run ends before the joint stage, early stopping never activates.

Stopping still saves `last.pt` and preserves `best.pt`. `summary.json` records why training ended; history and checkpoints record the patience counter. Compatible resumed checkpoints restore that counter. Older checkpoints without it start a fresh counter. Changing the patience or minimum improvement resets the counter deliberately; an already-stopped checkpoint will ask you to change/disable the policy before continuing.

Changes do not affect a process already running. Let the current epoch finish so `last.pt` exists, then restart with `--resume` if you want the new controls without losing completed epochs.

### Resume

```bash
train-window-frame --config src/window_frame/ai_guide/settings/training.toml \
  --resume output/window_frame/ai_guide/training/RUN/checkpoints/last.pt
```

Increase `epochs` beyond the last completed epoch. Keep dataset, model, stage lengths, batch size, accumulation, and seed unchanged. Resume restores model weights, optimizer progress, and random states at an epoch boundary. It creates a new linked run with its own best checkpoint, leaving the parent run untouched. Bit-for-bit reproducibility across different devices/software is not promised.

If interrupted mid-epoch, `last.pt` resumes from the last **completed** epoch. No checkpoint exists until the first epoch finishes.

**Only load `.pt` files you trust.** These checkpoints include Python training state, not just weights.

## 5. See learning progress

The terminal shows live progress bars for dataset checking, each training epoch, and validation, including completed/total counts, elapsed time, ETA, and running average loss where applicable. `run.log` keeps plain-text progress milestones every 30 seconds and at the start/end of each phase—no terminal-control characters. Redirected/non-interactive output uses those milestones instead of animated bars.

Every training run has:

- `run.log`: readable progress and failures.
- `history.json`: epoch losses, timing, and validation measurements.
- `loss.svg`: a chart readable in a browser, without extra dependencies.
- `visualizations/`: five distinct validation examples per epoch (or all available if fewer than five), each comparing input, target, prediction, and masks. Selection rotates reproducibly through the validation split; images repeat once the pool is exhausted. Filenames include the epoch and sample ID. TensorBoard also receives each selected comparison when enabled.
- `checkpoints/best.pt` and `last.pt`, each with a JSON metadata sidecar.

Training loss can jump when the training stage changes, because new objectives are introduced. Validation uses a fixed objective to keep checkpoint selection comparable.

For an interactive local dashboard:

```bash
uv sync --extra dashboard
```

Set `training.tensorboard = true`, then train. Open the dashboard with:

```bash
.venv/bin/tensorboard --logdir output/window_frame/ai_guide/training
```

Open the URL TensorBoard prints. Runs are shown separately so you can compare experiments. This is local; no external account or upload is required. Loss, accuracy, confidence-calibration measurements, and example images are recorded. There is no full TensorBoard dependency unless you enable it.

## 6. Evaluate a saved model

```bash
evaluate-window-frame \
  --checkpoint output/window_frame/ai_guide/training/RUN/checkpoints/best.pt
```

Use `--split validation` for development; reserve the test split for less frequent final comparisons. Repeatedly choosing settings against test results weakens the test's independence.

The evaluation saves overall and per-image results, plus `evaluation.svg`: a browser-readable chart of class accuracy and confidence reliability.

- Frame overlap and boundary accuracy.
- Accuracy for each rail class.
- False detections in photos without frames.
- Frame enhancement error compared with the desired target.
- The unchanged input's error, so you can see whether editing actually helped.
- Unwanted background and protected-region color changes.
- Confidence reliability: how closely confidence agrees with correct class predictions.
- Timing and peak GPU memory, when available.

Smaller image error and calibration error are better; larger frame overlap is better. Positive `frame_improvement` means the model beat doing nothing. Missing/undefined measurements are `null`, not a misleading zero. Boundary scores allow a one-pixel tolerance at the configured evaluation resolution.

Evaluation verifies the dataset against the original audit. Keep training run metadata alongside its checkpoints. Run summaries identify the checkpoint by its SHA-256 fingerprint (a file identity), not merely the filename. Compare runs using the same split, image size, and confidence threshold.

The current evaluator compares against your existing pipeline's **saved target images**. It does not execute the old pipeline, which is outside this package.

## 7. Enhance new photos

```bash
predict-window-frame \
  --checkpoint output/window_frame/ai_guide/training/RUN/checkpoints/best.pt \
  --input datasets/window_frame/test/00001.jpg
```

`--input` also accepts a folder of JPG, PNG, or WebP photos. `--confidence 0.9` raises the threshold to leave more uncertain regions unchanged. No mask or guide is needed as input.

Each photo gets its own folder containing:

| File | Meaning |
| --- | --- |
| `input.png` | Original, with EXIF orientation applied |
| `enhanced.png` | Lossless enhanced image |
| `frame_mask.png` | Black-and-white predicted frame pixels |
| `soft_mask.png` | Viewable frame probability, 0–255 |
| `soft_mask.npy` | Full-precision frame probabilities, 0–1 |
| `rail_classes.png` | Raw class IDs, 0–5 |
| `class_probabilities.npy` | Probability for each class, shape 6×height×width |
| `rail_guide.png` | Colored class overlay and legend |
| `class_confidence.png` | Red-to-green class certainty visualization |
| `edit_gate.npy` | Where edits were permitted and their blending strength |
| `comparison.png` | Input and enhanced image side by side |
| `metadata.json` | Source/checkpoint identity, timing, confidence summaries, warnings, and output hashes |

### Any photo dimensions—no fixed ratio or resolution list

A photo can be 3000×3129, 3102×2310, a narrow panorama, a portrait, or any other positive width and height that fits available memory. Dimensions are read separately from each photo; there is no list of allowed resolutions, required aspect ratio, or requirement that either side divide evenly by 512.

Training can stay at 512×512. By default, prediction processes overlapping sections of at most 512×512 pixels **without shrinking or stretching the photo**. That number is an internal memory-management limit, not a required photo size. Shared areas are blended before deciding which pixels to edit. The network accepts rectangular sections and odd dimensions directly. Only sides smaller than 16 pixels are padded for the network, then trimmed back to their original size. Images, masks, and raw class maps keep the original dimensions; the rail-guide canvas adds a legend below the image.

```bash
predict-window-frame --checkpoint PATH/TO/best.pt --input your_4k_photo.jpg
```

Only one section is sent to the GPU at a time. Larger photos take more time and CPU RAM, rather than requiring all model calculations at 4K on the GPU. Full-resolution probability exports can occupy hundreds of megabytes per 4K photo. The default overlap is 128 pixels for a 512-pixel training size; `--overlap 192` increases overlap and processing time.

For the previous faster approach, add `--mode resized`. This shrinks the entire photo to the training size and enlarges its predictions afterward, which may lose thin-frame details. The selected mode and overlap are recorded in the run and photo metadata.

Pixels excluded by the predicted edit gate are copied exactly from the decoded input. This does not guarantee protection of a real frame that the model misclassifies.

Tiled prediction sees only part of the scene at once. Blending reduces abrupt joins but cannot guarantee invisible seams or correct decisions. Evaluate it on real large photos: training on whole 512×512 scenes does not automatically teach every frame scale seen in a 4K crop. Future training examples should include native-resolution crops if quality suffers. The existing scored evaluator still uses resized training-resolution pairs, so its scores do not measure native-resolution tiled quality. The soft mask represents learned frame membership, not a guaranteed edge-transparency measurement. An image-wide average probability is not the probability that a frame exists. No honest automatic "this enhancement looks good" score is supplied.

The guide colors frame types; it does not assign separate object IDs such as P1/P2. Individual-rail numbering would require additional labeling/logic beyond this semantic class map.

## Output organization

```text
output/window_frame/ai_guide/
├── training/TIMESTAMP_UNIQUEID/
│   ├── metadata.json
│   ├── config.json
│   ├── dataset_audit.json
│   ├── dataset_split.json
│   ├── run.log
│   ├── history.json
│   ├── loss.svg
│   ├── dashboard/           # when enabled
│   ├── visualizations/
│   └── checkpoints/
│       ├── best.pt
│       ├── best.json
│       ├── last.pt
│       └── last.json
├── evaluation/TIMESTAMP_UNIQUEID/
└── predictions/TIMESTAMP_UNIQUEID/
    ├── metadata.json
    ├── settings.json
    ├── run.log
    ├── summary.json
    └── 00000_PHOTO_NAME/
```

Run metadata includes software versions, source-code fingerprints, and Git information. Different runs never overwrite each other. `last.pt` and `best.pt` are intentionally updated **within** one training run.

## Automated checks

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s src/window_frame/ai_guide/tests -v
```

These use temporary artificial examples, not full training on your photos. They cover input labels, unknown pixels, grouped splits, output shapes, learning a step, protection, checkpoint resumption, evaluation, and all prediction exports.
