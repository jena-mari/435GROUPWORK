# Reproduction guide

This guide distinguishes the compact outputs committed to GitHub from the
full local experiment directories used to produce them.

## 1. Verify the reported comparison without retraining

The committed prediction files are sufficient to reproduce the report tables.
From the repository root, install the evaluation environment and run:

```bash
python3.12 -m venv .venv-cnn
source .venv-cnn/bin/activate
python -m pip install --upgrade pip
python -m pip install -r data_preprocessing/requirements.txt \
  -r resnet18/requirements.txt -r evaluation/requirements.txt

python -m evaluation.role4_evaluation \
  --svm results/baseline \
  --frozen results/frozen_resnet18 \
  --finetuned results/finetuned_resnet18 \
  --audit Dhan-Shomadhan/preprocessing \
  --out /tmp/rice-comparison-rebuilt
```

This validates all five seeds for each method and recreates the nine rows in
`results/comparison/five_run_summary.csv`. The output contains per-run,
per-class, background-specific and combined metrics. The compact result folders
do not contain model checkpoints.

## 2. Re-run the models from the original images

Download version 1 of Dhan-Shomadhan from the Mendeley Data record and extract
it without renaming the original folders. Set paths from the repository root:

```bash
REPO_DIR=/path/to/435GROUPWORK
DATASET_DIR=/path/to/Dhan-Shomadhan
```

Verify the source images and manifests:

```bash
python data_preprocessing/verify_dataset.py \
  --root "$DATASET_DIR" \
  --audit "$REPO_DIR/Dhan-Shomadhan/preprocessing"
```

### HSV–LBP–SVM

The SVM uses its own environment because OpenCV requires NumPy below 2.3:

```bash
python3.12 -m venv .venv-svm
source .venv-svm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r hsv_lbp_svm/requirements.txt

python -m hsv_lbp_svm.svm_baseline \
  --dataset "$DATASET_DIR" \
  --audit "$REPO_DIR/Dhan-Shomadhan/preprocessing" \
  --output "$DATASET_DIR/experiments/baseline" \
  --seeds 11 22 33 44 55
deactivate
```

### Frozen ResNet-18

Activate `.venv-cnn`, install the CNN requirements, and cache the verified
ImageNet weights:

```bash
source .venv-cnn/bin/activate
python -m resnet18.check_cnn download-weights --dataset "$DATASET_DIR"
```

The reported frozen runs use the official directory
`$DATASET_DIR/experiments/frozen_resnet18_8ep/` and an eight-epoch cap:

```bash
python -m resnet18.train_frozen \
  --dataset "$DATASET_DIR" \
  --audit "$REPO_DIR/Dhan-Shomadhan/preprocessing" \
  --output "$DATASET_DIR/experiments/frozen_resnet18_8ep" \
  --device mps --workers 4 --epochs 8 --validation-only

python -m resnet18.train_frozen \
  --dataset "$DATASET_DIR" \
  --audit "$REPO_DIR/Dhan-Shomadhan/preprocessing" \
  --output "$DATASET_DIR/experiments/frozen_resnet18_8ep" \
  --device mps --workers 4 --evaluate-only
```

Use `--device cpu` on a machine without Apple MPS. This can produce small
backend-dependent numerical differences.

### Fine-tuned ResNet-18

The reported fine-tuned runs use the selected frozen checkpoint from the same
seed and the official directory
`$DATASET_DIR/experiments/finetuned_resnet18_4ep/`:

```bash
python -m resnet18.train_finetuned \
  --dataset "$DATASET_DIR" \
  --audit "$REPO_DIR/Dhan-Shomadhan/preprocessing" \
  --frozen "$DATASET_DIR/experiments/frozen_resnet18_8ep" \
  --output "$DATASET_DIR/experiments/finetuned_resnet18_4ep" \
  --device mps --workers 4 --epochs 4 --validation-only

python -m resnet18.train_finetuned \
  --dataset "$DATASET_DIR" \
  --audit "$REPO_DIR/Dhan-Shomadhan/preprocessing" \
  --frozen "$DATASET_DIR/experiments/frozen_resnet18_8ep" \
  --output "$DATASET_DIR/experiments/finetuned_resnet18_4ep" \
  --device mps --workers 4 --evaluate-only
```

### Final evaluation

After all three methods have produced test predictions, use the official
comparison directory:

```bash
python -m evaluation.role4_evaluation \
  --svm "$DATASET_DIR/experiments/baseline" \
  --frozen "$DATASET_DIR/experiments/frozen_resnet18_8ep" \
  --finetuned "$DATASET_DIR/experiments/finetuned_resnet18_4ep" \
  --audit "$REPO_DIR/Dhan-Shomadhan/preprocessing" \
  --out "$DATASET_DIR/experiments/comparison_8ep_4ep"
```

The five runs are repeated group-preserving holdouts, not five-fold
cross-validation. Their test partitions may overlap between seeds. Raw images,
checkpoints and feature caches are deliberately excluded from GitHub; the
source dataset and ImageNet weights must be supplied locally for retraining.
