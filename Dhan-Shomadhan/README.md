# Dataset and generated outputs

The original leaf images are shared inputs to all three methods. Keep their
folder names and filenames unchanged because the saved manifests use those paths.
Download instructions are in the [project README](../README.md).

| Directory or file | Used by | Contents |
|---|---|---|
| `Field Background  /` and `White Background /` | All methods | Original full-resolution images, downloaded locally |
| `Dhan-Shomadhan_picture_Information.csv` | Shared preparation | Supplied annotation metadata |
| `preprocessing/` | All methods | Audit records, review decisions, fixed mappings and shared split manifests |
| `preprocessing/figures/` | Dataset/report preparation | Sample-image figure, candidate-pair sheets and class contact sheets |
| `preprocessing/thumbnails/` | Audit review only | Small image previews used by the review pages and figure generator |
| `svm_cache/` | HSV–LBP–SVM | Local fixed-feature cache; scalers are fitted separately for each run |
| `experiments/baseline/` | HSV–LBP–SVM | Per-seed selected models, predictions, configurations and metrics |
| `experiments/baseline_validation/` | HSV–LBP–SVM | Validation-only checks; no test scores |
| `experiments/frozen_resnet18/` | Frozen ResNet-18 | Selected `best.pt`, histories and predictions from the integrated frozen experiment |
| `experiments/finetuned_resnet18/` | Fine-tuned ResNet-18 | Full-run CNN files after the Role 6 loop is integrated |
| `experiments/resnet18_checks/` | CNN development | Small training/validation checks and input-policy previews; not final results |
| `experiments/comparison/` | Evaluation/analysis | Five-run comparisons, per-class metrics, prediction review and confusion matrices |
| `model_cache/torchvision/` | Both CNN variants | Official pretrained weight files, cached locally |

The raw-image, cache and experiment directories are local and may not appear in
a fresh clone. Original images are not included in Git. Preprocessing figures
are inspection material, not replacement training images.

Read [`preprocessing/CURRENT_SPLIT_VERSION.txt`](preprocessing/CURRENT_SPLIT_VERSION.txt)
to find the shared split version. Its `seed_<seed>/` directories each contain
`train.csv`, `validation.csv` and `test.csv`. All methods use those same files.
The reorganisation does not alter their image membership or checksums.

Future model-generated images belong under
`experiments/<method>/figures/`, beside the relevant run records. Use separate
filenames for different seeds and background scenarios. Small selected results
can be added explicitly; checkpoints and feature caches should remain local.

The [shared CNN components](../resnet18/README.md) use the same manifests and
store preprocessing/seed identity in checkpoints. Diagnostic checkpoints are
marked separately and cannot be used as full-experiment checkpoints.

The [frozen experiment guide](../resnet18/ROLE5_FROZEN_EXPERIMENTS.md) and
[evaluation guide](../evaluation/README.md) explain execution and handover.
Role 4's supplied thumbnail results are separately archived in
`evaluation/reference/role4_thumbnail_svm/` at the repository root; they are
preliminary and do not replace the full-resolution baseline.

The [shared CNN component guide](../resnet18/SHARED_COMPONENTS_INTEGRATION.md)
includes a reproducible transformation preview from original training images.
Selected preview/check records are shared under `resnet18/validation_reports/`;
large diagnostic checkpoints remain local.
