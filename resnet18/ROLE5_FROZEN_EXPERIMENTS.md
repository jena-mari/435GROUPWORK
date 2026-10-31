# Frozen ResNet-18 experiment

`train_frozen.py` integrates Role 5's supplied training loop with the current
`resnet18/common.py`. Run commands from the repository root in `.venv-cnn`.
The original images can stay outside the repository; pass their dataset directory
with `--dataset`. The shared audit defaults to the one in this repository.

## Start with a small check

```bash
source .venv-cnn/bin/activate
python -m resnet18.train_frozen --dataset /path/to/Dhan-Shomadhan \
  --seeds 11 --smoke --epochs 1 --batch-size 4
```

This verifies the source-image and manifest checksums, then uses ten training
and ten validation images, covering the ten disease/background combinations.
It never evaluates test images. Outputs go to
`experiments/resnet18_checks/frozen_smoke/seed_11/` under the dataset.
These checkpoints are marked as implementation checks and cannot initialise
full-experiment fine-tuning or enter the final evaluation.

## Train the five full-data runs

```bash
python -m resnet18.train_frozen \
  --dataset /path/to/Dhan-Shomadhan \
  --output /path/to/Dhan-Shomadhan/experiments/frozen_resnet18_8ep \
  --device mps --workers 4 --epochs 8 --validation-only
```

The reported run uses seeds 11, 22, 33, 44 and 55; Adam with head learning rate
`1e-3`; batch size 32; an 8-epoch cap; and patience 5. The original planned
maximum was 30 epochs. Only the 2,565 head parameters
are updated. The backbone stays in evaluation mode, including BatchNorm running
statistics. The loop reapplies that policy before each training epoch and checks
that only the head changed. Cross-entropy receives raw five-class logits.

The selected checkpoint has the strictly highest validation macro-F1. Exact ties
count as no improvement. Training, validation and test membership comes from the
shared manifests; no new splits are generated. The agreed input policy is
224 x 224 aspect-preserving padding, ImageNet normalisation, training-only
horizontal flips and expanded rotations up to 10 degrees. There is no colour
jitter, random crop or Gaussian blur. See [COMMON_COMPONENTS.md](COMMON_COMPONENTS.md).

The command above trains and selects without test evaluation. Once the training
procedure is fixed, export the selected models' test predictions with:

```bash
python -m resnet18.train_frozen \
  --dataset /path/to/Dhan-Shomadhan \
  --output /path/to/Dhan-Shomadhan/experiments/frozen_resnet18_8ep \
  --device mps --workers 4 --evaluate-only
```

Alternatively, omit `--validation-only` when first training to train, select and
evaluate each run in one command. `--evaluate-only` uses saved settings and
checkpoints; it does not retrain. It requires matching manifests and input policy.

## Outputs and handover

Reported full-run directory: `experiments/frozen_resnet18_8ep/seed_<seed>/`.

| File | Use |
|---|---|
| `run_config.json` | Actual settings, image policy, versions, code hashes, selected epoch and frozen-backbone check |
| `validation_history.csv` | Training and validation loss/accuracy and validation macro-F1 per epoch |
| `best.pt` | Selected frozen checkpoint; keep local and pass to Role 6 |
| `validation_predictions.csv` and `.metadata.json` | Selected-checkpoint validation predictions |
| `validation_metrics.json` | Validation metrics, clearly separate from test results |
| `test_predictions.csv` and `.metadata.json` | Test predictions for Role 4, created only during test evaluation |
| `metrics.json` and `test_evaluation.json` | Field, white and combined test metrics; checkpoint identity |

A single invocation evaluating all five seeds also writes `summary.csv` in the
method directory, using mean and sample standard deviation. Separately completed
seeds can be combined with the [evaluation package](../evaluation/README.md).

For Role 6, use `initialise_finetuned(path_to_best_pt, data.context, device)` or
run the integrated `resnet18.train_finetuned` driver described in
`REPRODUCTION.md`.
The helper verifies seed, all three manifest hashes, policy and experiment scope.
Start a new fine-tuning optimiser and checkpoint selector. The full fine-tuned
experiment driver is integrated as `resnet18/train_finetuned.py`; use the
commands in [REPRODUCTION.md](../REPRODUCTION.md) for the reported run.

Use `--output` for a new experiment directory under the dataset. Existing runs
are never overwritten. Options include `--device cpu`, `--workers`,
`--weights-path` for the official offline weights, and recorded overrides for
`--epochs`, `--batch-size` and `--patience`. The learning rate and input policy
remain fixed by the common components. Small checks and altered settings must
not be presented as the planned final five-run experiment.

## Changes made during integration

The supplied folder included an alternative common implementation with portrait
448 x 224 inputs, stronger augmentation and a validation-loss tie-breaker.
The integrated loop uses the already agreed common implementation instead. This
keeps both CNN variants on the same input policy and checkpoint format. The
supplied destructive overwrite option was replaced with fresh output directories.
The completed compact frozen-model outputs are in `../results/frozen_resnet18/`.
The corresponding fine-tuned outputs are in `../results/finetuned_resnet18/`.
