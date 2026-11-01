# Submission user manual

Place `group_name.py` beside `Dhan-Shomadhan/` and run commands from that
directory. No cloning is required. In the existing repository, replace
`group_name.py` with `CSCI435Group2/group_name.py` below.

## Installation and dataset

Use Python 3.12 (the recorded experiments used 3.12.14). Download Dhan-Shomadhan
v1 from https://data.mendeley.com/datasets/znsxdctwtt/1 and preserve its original
folder spelling and trailing spaces. The default dataset path is
`./Dhan-Shomadhan/`; `--dataset /absolute/path` is also supported.

| Task | Command / explanation |
|---|---|
| Export runtime and original manifests | `python3.12 group_name.py setup` |
| Install both environments | `python3.12 group_name.py install` |
| Check images and original splits | `python3.12 group_name.py prepare` |
| Obtain official ImageNet weights | `python3.12 group_name.py download-weights` |
| Check loader, weights, parameter updates and checkpoints | `python3.12 group_name.py check` |

`install` creates separate environments under `Dhan-Shomadhan/.submission_envs/`.
SVM uses NumPy 2.2.6, Pillow 12.3.0,
OpenCV 4.12.0.88, SciPy 1.16.3, scikit-image 0.25.2 and scikit-learn 1.7.2.
CNN/evaluation uses NumPy 2.3.5, Pillow 12.3.0, SciPy 1.16.3,
scikit-learn 1.7.2, Matplotlib 3.10.7, PyTorch >=2.4 and ReportLab >=4.0.
Existing `.venv-svm` and `.venv-cnn` environments are detected automatically;
alternatively specify `--svm-python` and `--cnn-python`. No activation or shell
variables are required.

`prepare` verifies all 1,106 images, the metadata and the original 15 manifests;
it does not regenerate splits or replace existing audit records. Expected output
begins `PASS: all 1106 source images`. Official weights are downloaded to
`Dhan-Shomadhan/pretrained/resnet18-f37072fd.pth` and checked on loading.
Justify ImageNet pretraining in the report's Problem Description and Method:
it supplies initial visual features, with higher-level adaptation selected on validation data.

## Reproducing the report

| Task | Command |
|---|---|
| Recompute all three methods' recorded metrics, without images or retraining | `python3.12 group_name.py evaluate-reported` |
| Train all three methods on seeds 11, 22, 33, 44, 55 | `python3.12 group_name.py train` |
| Compare the latest newly trained models | `python3.12 group_name.py evaluate` |

`train` creates a fresh `Dhan-Shomadhan/experiments/submission-*` directory,
runs the full SVM grid, frozen ResNet-18 (maximum 30 epochs) and both fine-tuning
candidates (maximum 20 epochs each). All use the original five group-preserving
holdouts. Selection uses validation macro-F1 and patience 5. Evaluation prints
the path to `five_run_summary.csv`, containing means and sample SDs for field,
white and combined images. Hardware/software can affect new training;
`evaluate-reported` reproduces the recorded metrics directly.

## Additional repository experiments

| Task | Command |
|---|---|
| Individual training stages | `train-svm`, `train-frozen`, `train-cnns`, `train-finetuned` |
| Preview transformations | `python3.12 group_name.py preview` |
| Short validation-only training check | `python3.12 group_name.py smoke --epochs 1 --limit 8 --cpu` |
| Portrait, squash and padding comparison on validation data | `python3.12 group_name.py compare-resize --epochs 10` |
| Re-audit into a separate directory without replacing report manifests | `python3.12 group_name.py audit` |
| Recompute supplied preliminary thumbnail results separately | `python3.12 group_name.py evaluate-thumbnail` |
| Retrain the fixed-C preliminary thumbnail SVM | `python3.12 group_name.py train-thumbnail` |

Prepend `python3.12 group_name.py` to individual-stage names. Stages share the
latest training directory; `--output` selects another and `evaluate --runs`
scores it. Thumbnail controls and diagnostics remain separate from the final
comparison. `--cpu` forces CPU; `--workers 0` is the portable default.
All generated files stay under the dataset. Submit only the `.py` and report
`.pdf`; exclude images, generated folders, environments and checkpoints.
