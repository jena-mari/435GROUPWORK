# Group handover

The repository now contains the shared data preparation, the full SVM run,
five frozen ResNet-18 runs, five fine-tuned ResNet-18 runs and their comparison.
Use the saved predictions in `results/` when writing the report.

## Who uses what

| Work | Location | Handover |
|---|---|---|
| Dataset audit and splits | `data_preprocessing/`, `Dhan-Shomadhan/preprocessing/` | Keep the current split version unchanged |
| HSV–LBP–SVM | `hsv_lbp_svm/`, `results/baseline/` | Full validation grid; scaler fitted on training data only |
| Frozen ResNet-18 | `resnet18/train_frozen.py`, `results/frozen_resnet18/` | ImageNet backbone, head only; 8-epoch actual cap |
| Fine-tuned ResNet-18 | `resnet18/train_finetuned.py`, `results/finetuned_resnet18/` | Same-seed frozen checkpoint, `layer4` + head; 4 additional epochs |
| Evaluation | `evaluation/`, `results/comparison/` | Checks IDs and manifests, then produces tables and matrices |

## Result files

Each method has five directories, one per seed. Keep these small files when
sharing results:

- `run_config.json`
- `validation_history.csv`
- `test_predictions.csv`
- `metrics.json`
- `test_evaluation.json`

The comparison folder contains `five_run_summary.csv`, `per_run_metrics.csv`,
`per_class_metrics.csv`, `prediction_review.csv`, `paired_differences.csv` and
the confusion matrices. Do not commit `best.pt`, `selected_model.pkl`, SVM
feature caches or raw images.

## Report values

Use the mean and sample standard deviation columns from
`results/comparison/five_run_summary.csv`. Report white, field and combined
test subsets separately. Combined scores are calculated from all predictions,
not by averaging the two background scores.

For the discussion, use `per_class_metrics.csv` to identify class-level patterns
and `prediction_review.csv` to choose representative images. Check the original
images before explaining why a prediction failed. Treat background effects as
observations from this dataset, not as causal proof.

## Re-run checks

```bash
python data_preprocessing/verify_dataset.py \
  --root /path/to/Dhan-Shomadhan \
  --audit Dhan-Shomadhan/preprocessing
python -m resnet18.check_cnn check --dataset /path/to/Dhan-Shomadhan
python -m evaluation.role4_evaluation \
  --svm results/baseline --frozen results/frozen_resnet18 \
  --finetuned results/finetuned_resnet18 --out /tmp/comparison-check
```

The compact results in GitHub were generated from the full local dataset. The
source images and checkpoints must be downloaded or regenerated locally when a
new experiment is needed.
