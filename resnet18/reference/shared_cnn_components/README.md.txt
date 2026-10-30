# ResNet-18 methods

This folder is reserved for the frozen and partly fine-tuned ResNet-18
implementations. **The shared CNN components and checks are in place
(`common.py`, `check_cnn.py`); the frozen and fine-tuned training scripts, CNN
figures and results have not been added yet.** The settings below are the
project plan, not completed runs.

Both variants must read the same `RiceDataset` loader, disease mapping and
current manifest version used by the HSV–LBP–SVM baseline. The CNN input
transform, normalisation and training augmentation are defined once in
`common.py` and passed through `RiceDataset(transform=...)`. SVM resizing and
blurring are specific to that method and are not used here.

| Variant | Starting checkpoint | Trainable parameters | Planned settings | Local output directory |
|---|---|---|---|---|
| Frozen | ImageNet-pretrained ResNet-18 | Replacement five-class linear head | Adam, head LR `1e-3`, batch size 32, up to 30 epochs | `Dhan-Shomadhan/experiments/frozen_resnet18/` |
| Fine-tuned | Best frozen checkpoint from the same seed | `layer4` and the five-class head | Adam, `layer4` LR `1e-5`, head LR `1e-4`, batch size 32, up to 20 additional epochs | `Dhan-Shomadhan/experiments/finetuned_resnet18/` |

Use validation macro-F1 for checkpoint selection, with five consecutive epochs
without improvement as the stopping rule. Keep the agreed BatchNorm running
statistics fixed, including after calls to `model.train()`. Evaluate test images
only after the training and selection procedure is fixed.

Each variant needs runs for seeds 11, 22, 33, 44 and 55. For every seed, return
the configuration, validation history, selected checkpoint location and test
predictions in the format described in the [group handover](../GROUP_HANDOVER.md).
Keep model checkpoints and temporary arrays local.

Future learning curves, confusion matrices or feature visualisations should go
under that variant's `Dhan-Shomadhan/experiments/<method>/figures/` directory and
identify their seed, background subset and source run. They should not be mixed
with preprocessing sample images or SVM figures.

The [shared preprocessing guide](../data_preprocessing/README.md) explains the
loader and splits. Preserve the current partitions when adding the CNN code.

## Shared components (`resnet18/common.py`)

Both training scripts should import these rather than copying them, so that the
two variants differ only in which layers are trained.

| Component | What it provides |
|---|---|
| `load_shared_splits(root, audit, seed)` | `RiceDataset` objects for the current version's `train.csv`, `validation.csv` and `test.csv`, with the CNN transforms attached. Repeats the baseline's checks: manifest checksums, seed/partition fields, groups kept within one partition, full coverage. |
| `TrainTransform(seed)`, `EvalTransform()` | CNN-specific transforms for `transform=`. Input is the loader's RGB PIL image; output is a normalised float tensor. |
| `make_loader(dataset, train, seed, workers=0)` | Seeded shuffling for training; fixed order otherwise. Batches are dicts with `image`, `disease_id`, `background_id` and `image_id`. Training augmentation is reseeded per worker, so it is reproducible for a given seed and worker count. |
| `build_resnet18(root)` | ResNet-18 in plain PyTorch (no torchvision). ImageNet weights are checked against PyTorch's published SHA-256 prefix, every backbone tensor must load, and the 1,000-class head is replaced by a new 512 → 5 layer. |
| `set_trainable(model, {stage: lr})` | Freezes everything, then unfreezes the named stages and returns Adam parameter groups. Frozen: `{"fc": 1e-3}`. Fine-tuned: `{"layer4": 1e-5, "fc": 1e-4}`. |
| `ResNet18.train()` | Keeps every BatchNorm layer in eval mode after `model.train()`, so running statistics stay at their ImageNet values (the agreed policy). BatchNorm scale/shift in unfrozen stages still learn. |
| `BestTracker`, `save_checkpoint`, `load_checkpoint` | Highest validation macro-F1 (exact ties keep the lower validation loss), stop after five epochs without improvement. Checkpoints store the class mapping and run metadata. |
| `predict`, `check_predictions`, `write_predictions`, `get_scores`, `summary_rows` | Predictions keyed by `image_id`; checks for missing, repeated or extra IDs; `test_predictions.csv` in the handover format with `prob_0`–`prob_4`; `metrics.json` and summaries in the same layout as the baseline. |
| `new_run_dir`, `base_run_config` | `Dhan-Shomadhan/experiments/<method>/seed_<seed>/` and the shared `run_config.json` fields (manifest version and checksums, transforms, weight source and hash, selection rule, BatchNorm policy, package versions, device, determinism settings). |

## CNN transforms

1,089 of the 1,106 images are 1952 × 4160 portrait photographs, 11 are the same
size in landscape, and 6 are 1536 × 3264. Squashing them to 224 × 224 would
stretch leaves about twice as wide; padding them would shrink the whole
photograph to about 105 pixels wide. The proposed default (`RESIZE_MODE = "portrait"`) turns the
landscape images upright and resizes every image to **448 × 224 (height ×
width)**. This keeps the leaf's shape and gives small lesions twice the pixels
of a square input. ResNet-18's global average pooling accepts this input size
without architectural changes.

| Step | Training | Validation / test |
|---|---|---|
| Orientation | Landscape images turned upright | Same |
| Crop | Random region covering 60–100% of the image, aspect ratio jittered by 3:4–4:3 | None (whole image) |
| Resize | 448 × 224, bilinear | 448 × 224, bilinear |
| Flips | Horizontal and vertical, each p = 0.5 | None |
| Rotation | Uniform ±15°, corners filled with the ImageNet mean colour | None |
| Brightness / contrast | Each ×(1 ± 0.2) | None |
| Hue / saturation | Unchanged, because lesion colour is diagnostic | Unchanged |
| Normalisation | ImageNet mean and standard deviation (fixed, not fitted to our data) | Same |

`squash` (224 × 224) and `pad` (224 × 224 with mean-colour padding) are also
available. `python -m resnet18.check_cnn compare-resize` trains the frozen head
for each option on seed 11 and compares **validation** macro-F1 only. The group
should fix the mode before running the five reported seeds and record any change
in `common.py`, since both variants must use the same policy.

## Setup and checks

```bash
python -m pip install -r resnet18/requirements.txt
python data_preprocessing/verify_dataset.py
python -m resnet18.check_cnn download-weights      # -> Dhan-Shomadhan/pretrained/resnet18-f37072fd.pth
python -m resnet18.check_cnn check                 # loader, transforms, weights, freezing, BatchNorm, export
python -m unittest discover -s resnet18/tests -t . -v
python -m resnet18.check_cnn preview               # experiments/resnet18_checks/figures/transform_preview_seed_11.png
python -m resnet18.check_cnn smoke --epochs 3      # short head-only run on seed 11, validation only
python -m resnet18.check_cnn compare-resize        # choose the resize mode on validation data
```

If the weights cannot be downloaded automatically, download
<https://download.pytorch.org/models/resnet18-f37072fd.pth> and place it, without
renaming, at `Dhan-Shomadhan/pretrained/resnet18-f37072fd.pth`.

The original JPEGs are large, and the shared loader fully decodes each one, so
data loading dominates CPU training time. Use `--workers 4` (or `workers=4` in
`make_loader`) to decode in parallel. On Windows, keep the training entry point
under `if __name__ == "__main__":`.

The check outputs under `experiments/resnet18_checks/` are implementation
checks, not reported experiments. They never use test images.
