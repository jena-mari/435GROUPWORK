## VII. Appendix: User Manual

The submitted `group_name.py` contains the combined implementation, shared
splits and recorded predictions. No repository clone or separate source files
are required. Run all commands from the folder containing `group_name.py`.

### VII.I Installation

| Item | Instructions |
|---|---|
| Python | Use Python 3.12; the recorded experiments used 3.12.14. |
| Dataset | Download [Dhan-Shomadhan v1](https://data.mendeley.com/datasets/znsxdctwtt/1). Place its `Dhan-Shomadhan/` folder beside `group_name.py`, including the metadata CSV. Preserve original spelling and trailing spaces: two after `Field Background` and one after `White Background`. Retain any existing `preprocessing/`. |
| Install packages | `python3.12 group_name.py install` |
| Package requirements | SVM: NumPy 2.2.6, Pillow 12.3.0, OpenCV 4.12.0.88, SciPy 1.16.3, scikit-image 0.25.2 and scikit-learn 1.7.2. CNN/evaluation: NumPy 2.3.5, Pillow 12.3.0, SciPy 1.16.3, scikit-learn 1.7.2, Matplotlib 3.10.7, PyTorch >=2.4 and ReportLab >=4.0. |
| Pretrained weights | `python3.12 group_name.py download-weights` |

The installer creates separate SVM and CNN environments inside
`Dhan-Shomadhan/.submission_envs/`. Commands select them automatically; no
activation is required. Official ImageNet ResNet-18 weights are downloaded to
`Dhan-Shomadhan/pretrained/resnet18-f37072fd.pth` and verified on loading.

### VII.II Verification

| Task | Command / expected result |
|---|---|
| Restore shared audit and verify dataset | `python3.12 group_name.py prepare` |
| Expected result | `PASS: all 1106 source images, source metadata and 15 manifests match the audited checksums.` |
| Check CNN implementation | `python3.12 group_name.py check` |

`prepare` preserves the original five splits rather than regenerating them.
Existing audit files are retained; conflicting files cause a clear error.

### VII.III Reproducing Results

| Task | Command |
|---|---|
| Recompute reported metrics without retraining | `python3.12 group_name.py evaluate-reported` |
| Retrain all three models | `python3.12 group_name.py train` |
| Compare the latest newly trained models | `python3.12 group_name.py evaluate` |

`evaluate-reported` requires neither source images nor pretrained weights.
For retraining, complete installation, dataset verification and weight download
first. Training runs the full HSV-LBP-SVM grid, frozen ResNet-18 and both partial
fine-tuning candidates using seeds 11, 22, 33, 44 and 55. CNN checkpoint selection
uses validation macro-F1 and five-epoch patience, with maximum epoch counts of
30 for frozen training and 20 per fine-tuning candidate.

Generated files remain under `Dhan-Shomadhan/`. Training creates a fresh
`experiments/submission-*` folder; evaluations create fresh `experiments/evaluation-*`
folders and print the path to `five_run_summary.csv`. Results include five-run
means and sample standard deviations for white, field and combined test images.
Fresh training can vary across hardware/software; supplied predictions reproduce
the recorded metrics directly.

### VII.IV Additional Experiments

Append the following command names to `python3.12 group_name.py`:

| Experiment | Command name / options |
|---|---|
| Individual model stages | `train-svm`, `train-frozen`, `train-finetuned`, `train-cnns` |
| Transform preview | `preview` |
| Small training check | `smoke --epochs 1 --limit 8 --cpu` |
| Portrait/squash/padding comparison | `compare-resize --epochs 10` |
| Separate dataset re-audit | `audit` |
| Preliminary thumbnail control | `train-thumbnail` or `evaluate-thumbnail` |

Diagnostics and thumbnail controls remain separate from the final comparison.
Use `--cpu` to force CPU execution; `--workers 0` is the default. Submit only
`group_name.py` and the final report PDF, excluding images and generated folders.
