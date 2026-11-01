# Person 6: Fine-tuned ResNet-18 experiments

This submission follows the same project-root layout as Person 5's archive.
Extract the main ZIP into a new folder or merge it with a copy of the shared
project. Commands run from the directory containing `resnet18/`,
`data_preprocessing/`, and `Dhan-Shomadhan/`.

## Completed outputs

All five seeds (11, 22, 33, 44, 55) were completed. The combined mean test
accuracy is 0.6662 and macro-F1 is 0.6317. These are repeated holdouts, not
five-fold cross-validation.

| Location from project root | Contents |
|---|---|
| `resnet18/finetune.py` | Person 6 training, validation selection and test export |
| `resnet18/run_role6.py` | Five-seed controller, including missing frozen prerequisites |
| `resnet18/package_role6.py` | Verify results and build the submission |
| `resnet18/export_role6_layout.py` | Export the Role 5-style directory and ZIPs |
| `resnet18/tests/test_finetune.py` | Gradient, BatchNorm, checkpoint and test-isolation checks |
| `Dhan-Shomadhan/preprocessing/` | Shared audit, label mapping and saved split manifests |
| `Dhan-Shomadhan/experiments/finetuned_resnet18/seed_<seed>/` | Run configuration, candidate histories, validation predictions, test predictions and metrics |
| `Dhan-Shomadhan/experiments/finetuned_resnet18/figures/` | Learning curves and test confusion matrices |
| `Dhan-Shomadhan/experiments/finetuned_resnet18/logs/` | Training, environment, data verification and test logs |
| `Dhan-Shomadhan/experiments/finetuned_resnet18/ROLE6_REPORT.pdf` | Readable report |
| `Dhan-Shomadhan/experiments/finetuned_resnet18/ROLE6_REPORT.md` | Editable report |
| `Dhan-Shomadhan/experiments/finetuned_resnet18/summary.csv` | Five-run means and sample standard deviations |
| `Dhan-Shomadhan/experiments/frozen_resnet18/seed_<seed>/` | Locally generated frozen prerequisite results and configurations |

Person 5's supplied archive contained code but no trained models. Its
`train_frozen.py` was used unchanged to generate the same-seed frozen starting
models. Coordinate these prerequisites with Person 5's final results. This
submission does not claim that those completed models were supplied by Person 5.

## Selected models

Model weights are in the separate `CSCI435_Role6_Selected_Checkpoints.zip`.
Extract that ZIP into the same project root as the main ZIP. Its paths match
the candidate checkpoint locations recorded in each run configuration:

`Dhan-Shomadhan/experiments/finetuned_resnet18/seed_<seed>/candidate_1/best.pt`

Checkpoints include the complete five-class model state and class mapping.
Original source images, ImageNet weights, and frozen checkpoint binaries are
downloaded or generated separately; the small prediction/configuration files
are included in the main submission.

## Reproduce

Use Python 3.12 and the original Dhan-Shomadhan V1 images, preserving paths in
the manifests. Install a GPU-enabled PyTorch build first for GPU training.
Recorded package versions are in the experiment's `requirements-used.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r resnet18/requirements.txt
.\.venv\Scripts\python.exe data_preprocessing/verify_dataset.py
.\.venv\Scripts\python.exe -m resnet18.check_cnn download-weights
.\.venv\Scripts\python.exe -m unittest discover -s resnet18/tests -t . -v
.\.venv\Scripts\python.exe -u -m resnet18.run_role6 --workers 6 --threads 4
.\.venv\Scripts\python.exe -m resnet18.package_role6
```

Keep completed result folders when inspecting the submission. For a fresh
training reproduction, use a separate project copy without previous experiment
outputs. The controller preserves completed stages and refuses to overwrite
incomplete runs. Reproducing the frozen prerequisites also requires their
checkpoint binaries; the included frozen CSV/JSON files are comparison records.

For Person 4, match test predictions on `image_id` and seed, not row order.
Disease IDs: brown_spot=0, leaf_scald=1, rice_blast=2, rice_tungro=3,
sheath_blight=4. Background IDs: field=0, white=1. The experiment root also
contains per-class F1, paired differences from frozen models, and failure IDs.
