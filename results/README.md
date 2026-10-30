# Experiment results

These are the compact outputs from the full local experiments. Each method has
five seed folders. The files are intentionally limited to configurations,
histories, predictions and metrics; checkpoints and feature caches stay local.

| Folder | Method | Actual training cap |
|---|---|---|
| `baseline/` | HSV–LBP–SVM | Full C/gamma validation grid |
| `frozen_resnet18/` | Frozen ImageNet ResNet-18 | 8 epochs maximum |
| `finetuned_resnet18/` | `layer4` + head from same-seed frozen checkpoint | 4 additional epochs maximum |
| `comparison/` | Three-method evaluation | Five runs per method |

The comparison table is `comparison/five_run_summary.csv`. It reports accuracy,
macro precision, macro recall and macro-F1 for white, field and combined test
images. Standard deviations are sample standard deviations over seeds 11, 22,
33, 44 and 55.

`comparison/prediction_review.csv` contains image IDs and source paths for
error inspection. Use the downloaded original images when looking at examples.
The pooled confusion figures under `comparison/figures/` count test appearances
from overlapping holdouts.
