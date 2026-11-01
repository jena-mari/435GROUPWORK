# Person 5: Frozen ResNet-18 experiments

## What this script does

`train_frozen.py` implements the frozen-backbone ResNet-18 experiment using the shared functions in `common.py`. It uses the current shared manifests for seeds 11, 22, 33, 44 and 55; trains only the five-class `fc` layer; selects the checkpoint by validation macro-F1 (ties use lower validation loss); and evaluates the test partition only after selecting the checkpoint.

For each seed, outputs are saved under:

`Dhan-Shomadhan/experiments/frozen_resnet18/seed_<seed>/`

Files include `best_checkpoint.pt`, `training_history.csv`, `test_predictions.csv`, `metrics.json`, and `run_config.json`. A five-run aggregate is written to `Dhan-Shomadhan/experiments/frozen_resnet18/summary.csv`.

## Setup and run

Run commands from the project root (the directory containing `data_preprocessing/`, `Dhan-Shomadhan/`, and `resnet18/`). Use Python 3.12 as specified in the assignment.

```bash
python -m pip install -r resnet18/requirements.txt
python data_preprocessing/verify_dataset.py
python -m resnet18.check_cnn download-weights
python -m resnet18.train_frozen
```

Optional quick smoke run on one split (not a reportable experiment):

```bash
python -m resnet18.train_frozen --seeds 11 --epochs 2 --overwrite
```

Other options include `--workers 4`, `--cpu`, and `--resize-mode portrait`. Keep the same resize mode for both CNN variants. Do not use `--overwrite` unless intentionally replacing the outputs for the selected seed(s).

## Important data requirement

The original dataset images must be present in the folders described by the shared manifests under `Dhan-Shomadhan/`. The repository ZIP may contain preprocessing thumbnails/contact sheets but not all original source images. The script will fail during data verification/loading if those original files are missing. Do not report performance numbers until the complete source dataset has been loaded and the five runs finish successfully.

## Implementation choices

- Pretrained ResNet-18 weights are loaded and verified through `common.py`.
- All backbone parameters are frozen; only `fc` is optimised using Adam at learning rate `1e-3`.
- Batch size is 32; maximum epochs 30; early-stopping patience 5.
- The shared transforms and BatchNorm policy are reused unchanged.
- Test data are never used for gradient updates or checkpoint selection.
- Predictions are exported in the shared handover format for Person 4.
