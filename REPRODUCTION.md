# Reproducing the results

Run commands from the repository root. Clone the repository only for first-time
setup; use `git pull --ff-only` to update an existing clean checkout.

## Installation

Use Python 3.12.14 and separate environments:

```bash
python3.12 -m venv .venv-svm
source .venv-svm/bin/activate
python -m pip install -r hsv_lbp_svm/requirements.txt
deactivate
python3.12 -m venv .venv-cnn
source .venv-cnn/bin/activate
python -m pip install -r data_preprocessing/requirements.txt -r resnet18/requirements.txt -r evaluation/requirements.txt
```

For training, download Dhan-Shomadhan v1 from
https://data.mendeley.com/datasets/znsxdctwtt/1. Merge its two source image
folders into the existing `Dhan-Shomadhan/`; preserve `preprocessing/` and
the original folder names, including trailing spaces. Do not replace the
entire dataset directory.

```bash
python data_preprocessing/verify_dataset.py --root Dhan-Shomadhan --audit Dhan-Shomadhan/preprocessing
python -m resnet18.check_cnn download-weights --dataset Dhan-Shomadhan
```

## Recompute the reported metrics without retraining

The committed predictions are sufficient; source images and model checkpoints
are unnecessary for this step. Use a fresh evaluation directory each time:

```bash
source .venv-cnn/bin/activate
EVAL_DIR=$(mktemp -d /tmp/rice-comparison-XXXXXX)
python -m evaluation.role4_evaluation --svm results/baseline --frozen results/frozen_resnet18 --finetuned results/finetuned_resnet18 --audit Dhan-Shomadhan/preprocessing --out "$EVAL_DIR/comparison"
```

The summary is `$EVAL_DIR/comparison/five_run_summary.csv`. Values are fractions;
multiply by 100 for percentages. Standard deviations use `ddof=1`.

## Full retraining

Complete dataset verification and weight installation first. Keep the following
commands in the same terminal. Set `RUN_DIR` once per complete reproduction;
all three models then write into this new directory. No folders need moving.

```bash
RUN_DIR=$(mktemp -d "$PWD/Dhan-Shomadhan/experiments/reproduction-XXXXXX")
source .venv-svm/bin/activate
python -m hsv_lbp_svm.svm_baseline --dataset Dhan-Shomadhan --audit Dhan-Shomadhan/preprocessing --output "$RUN_DIR/baseline" --seeds 11 22 33 44 55
source .venv-cnn/bin/activate
python -u -m resnet18.run_role6 --output "$RUN_DIR" --workers 0 --threads 4 --seeds 11 22 33 44 55
python -m evaluation.role4_evaluation --svm "$RUN_DIR/baseline" --frozen "$RUN_DIR/frozen_resnet18" --finetuned "$RUN_DIR/finetuned_resnet18" --audit Dhan-Shomadhan/preprocessing --out "$RUN_DIR/comparison"
```

The new summary is `$RUN_DIR/comparison/five_run_summary.csv`. The CNN runner
accepts `--cpu`, `--dataset` and `--audit` when needed. It preserves existing
outputs and rejects incomplete runs or supplied result bundles without training
checkpoints. To start over, create another `RUN_DIR`; to reuse a completed local
CNN training stage, retain its checkpoints and use its existing output path.

All methods use the five shared seeds, unchanged manifests, and validation-based
model selection. Fresh training may vary across hardware and PyTorch versions;
re-evaluating the supplied predictions reproduces the recorded metrics directly.
