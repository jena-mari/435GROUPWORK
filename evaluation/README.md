# Evaluation and comparative analysis

This package integrates Role 4's evaluation script with the shared manifests and
current SVM/CNN prediction format. It computes results from saved predictions;
it does not train models or choose checkpoints.

## Setup

Use the existing CNN environment or a separate Python 3.12 environment:

```bash
source .venv-cnn/bin/activate
python -m pip install -r evaluation/requirements.txt
python -m unittest discover -s evaluation/tests -v
```

Keep the OpenCV SVM training environment separate because it uses a different
NumPy version. Evaluation only needs the prediction CSVs, configurations and
shared audit; the original images are needed when visually reviewing examples.

## Compare the final methods

After each method has completed all five runs:

```bash
python -m evaluation.role4_evaluation \
  --svm /path/to/Dhan-Shomadhan/experiments/baseline \
  --frozen /path/to/Dhan-Shomadhan/experiments/frozen_resnet18 \
  --finetuned /path/to/Dhan-Shomadhan/experiments/finetuned_resnet18 \
  --out /path/to/Dhan-Shomadhan/experiments/comparison
```

Each source needs `seed_11/`, `seed_22/`, `seed_33/`, `seed_44/` and `seed_55/`,
with `test_predictions.csv` and `run_config.json`. Use the canonical SVM output
`experiments/baseline`, rather than the older `svm_results` example. You may
supply one or two methods while the others are pending; every supplied method
still needs all five seeds. The evaluation configuration marks a comparison as
complete only when all three methods are provided.

The evaluator checks manifest checksums, recorded seed/version/mapping, complete
image membership, duplicate IDs, labels, background IDs and method names before
writing outputs. CNN configurations must describe full experiments with ImageNet
initialisation; their input policies must agree. Diagnostic runs are rejected.
These checks verify saved records, not how an external training process used its
data. Retain the training code and configurations as supporting evidence.

Use a fresh `--out` directory; existing results are preserved. `--audit` selects
another shared audit location. `--no-plots` skips PNG generation.

## Results produced

| Output | Purpose |
|---|---|
| `per_run_metrics.csv` | Accuracy, macro precision, macro recall and macro-F1 by method, seed and background |
| `five_run_summary.csv` | Mean and sample SD (`ddof=1`) over five runs, calculated before rounding |
| `per_class_metrics.csv` | Precision, recall, F1 and support for each disease |
| `prediction_review.csv` | Correct/incorrect status, original path and identifiers for visual error analysis |
| `paired_differences.csv` | Seed-matched metric differences when multiple methods are supplied |
| `confusion_matrices.json` | Per-run and pooled matrices in the fixed five-class order |
| `figures/` | Confusion matrices for each seed and background, plus clearly labelled pooled matrices |
| `evaluation_config.json` | Source hashes, manifest identity, package versions, scope and metric conventions |

All CSV scores are fractions; multiply means and SDs by 100 for percentages.
White and field metrics are computed on their own subsets. Combined metrics use
the complete test predictions, rather than averaging the two background scores.
All macro metrics include all five diseases, with zero for undefined values.
Matrix rows are true classes and columns are predictions.

These are repeated holdouts with overlapping test sets. Pooled matrices count
test appearances, which may include the same image in several runs. They are
inspection figures, not an independent test set. Paired differences are descriptive;
the script does not infer statistical significance or causal explanations.

## Supplied preliminary thumbnail experiment

[reference/role4_thumbnail_svm/](reference/role4_thumbnail_svm/README.md) preserves
Role 4's supplied predictions, measured summaries, pooled figure, scripts and
PDF drafts. This experiment used preprocessing thumbnails and fixed SVM settings
`C=1`, `gamma=scale`. It is separate from the full-resolution HSV–LBP–SVM grid
search and both CNN experiments. Its scores must be labelled as preliminary.

Recheck those saved predictions separately:

```bash
python -m evaluation.role4_evaluation \
  --thumbnail-reference evaluation/reference/role4_thumbnail_svm \
  --out Dhan-Shomadhan/experiments/thumbnail_reference_check
```

This mode checks all five test manifests and recomputes metrics without training.
It cannot be combined with final-method sources. The historical thumbnail runner
has machine-specific paths and is preserved as a record, not a supported command.

## Continuing the report

For V.1, use actual saved configurations and list each completed experiment.
For V.2, use the five-run table for white, field and combined backgrounds. Include
mean and sample SD and identify any missing method; do not fill gaps using the
thumbnail results. For V.3, inspect original images using `prediction_review.csv`
and per-class matrices, then discuss representative successful, failed and
challenging cases. Keep explanations conditional unless supported by the images
or a controlled experiment. Method improvements should be selected using
validation data or tested as separately defined follow-up experiments.

The supplied PDF drafts use Section 6 numbering; the requested final report uses
V.1–V.3 and a two-page limit. Update the final report after all three full-resolution
methods have results. The reference PDFs remain unchanged.
