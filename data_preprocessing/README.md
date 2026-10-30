# Shared data preprocessing

This folder contains the preparation used by every method. It handles image
checks, folder-label mapping, similarity groups, shared splits and image loading.
Model-specific feature extraction and training belong in the method folders.

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Audit source images, generate shared holdouts, provide `RiceDataset` |
| `verify_dataset.py` | Check downloaded images and saved manifest checksums |
| `make_figures.py` | Rebuild sample figures, candidate-pair sheets and class contact sheets from the saved audit |
| `make_report.py` | Build the HTML inspection report and report-section text |
| `test_prepare_dataset.py` | Checks for grouping, coverage, source selection and loading |
| `requirements.txt` | Dependencies for the recorded preprocessing environment |
| `README.txt` | Detailed audit rules and reproduction instructions |

Run commands from the repository root:

```bash
python data_preprocessing/verify_dataset.py
python -m unittest discover -s data_preprocessing -p 'test_prepare_dataset.py' -v
python data_preprocessing/make_figures.py
```

The image-verification command needs the original dataset. Figure rebuilding
uses the committed inventory and thumbnails; it does not train a model or change
the shared partitions. Existing generated figure files are replaced when rebuilt.

Outputs are under [`Dhan-Shomadhan/preprocessing/`](../Dhan-Shomadhan/README.md):
audit CSV/JSON files, the current-version pointer, five split sets, thumbnails,
HTML review pages and the [`figures/`](../Dhan-Shomadhan/preprocessing/figures/README.md) folder.
Use the current saved splits for the comparison. A new audit must use a separate
output directory until the group agrees to adopt it.

The [project README](../README.md) covers dataset setup. The
[group handover](../GROUP_HANDOVER.md) covers training and results.

The CNN-specific policy and checkpoint guards are implemented in
[`resnet18/common.py`](../resnet18/common.py). The shared loader still returns
RGB PIL images unless a transform is supplied; the CNN factory supplies the
agreed tensor transform while keeping background and image IDs as metadata.

The integrated frozen training command and selected-checkpoint handover are in
[the frozen experiment guide](../resnet18/ROLE5_FROZEN_EXPERIMENTS.md).
[The evaluation package](../evaluation/README.md) reads each method's five saved
test prediction files against these same manifests; it does not regenerate
splits or train on inspection thumbnails.
