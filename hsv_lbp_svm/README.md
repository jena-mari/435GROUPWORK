# Running the SVM baseline

`hsv_lbp_svm/svm_baseline.py` contains the group-supplied colour/texture baseline, adapted to
read the shared manifests. Run it from the repository root after downloading
and verifying the source images as described in the [project README](../README.md).

## Folder contents

- `svm_baseline.py`: feature extraction, training, validation selection and evaluation.
- `requirements.txt`: the baseline environment's dependencies.
- `tests/test_svm_baseline.py`: shared-manifest and training-only-scaler checks.

The root `svm_baseline.py`, `requirements-svm.txt` and `test_svm_baseline.py`
files forward the older commands to this folder. Run the documented module
commands from the repository root so the shared preprocessing package is found.
Original source-image and manifest paths have not changed.

## Setup

Use Python 3.12 and a separate environment for the baseline:

```bash
python3.12 -m venv .venv-svm
source .venv-svm/bin/activate
python -m pip install -r hsv_lbp_svm/requirements.txt
python -m unittest discover -s hsv_lbp_svm/tests -p 'test_*.py' -v
```

On Windows, use `py -3.12 -m venv .venv-svm`, then activate with
`.venv-svm\Scripts\activate`. Preserve the dataset's original folder names;
the [project README](../README.md) explains the trailing-space issue.

The baseline uses NumPy 2.2.6 because OpenCV 4.12 requires NumPy below 2.3 on
Python 3.12. The original audit environment used NumPy 2.3.5. Keep these separate:
do not install both requirements files into one environment. The saved image
membership and label mapping are unchanged. The baseline also uses the shared
Pillow loader for EXIF orientation and RGB conversion before converting to
OpenCV's BGR format.

## Check one run first

```bash
python -m hsv_lbp_svm.svm_baseline --seeds 11 --validation-only
```

This extracts training/validation features, searches the parameter grid and saves
the selected model and validation history. It does not extract features for or
score the test images. By default, files go under
`Dhan-Shomadhan/experiments/baseline_validation/seed_11/`.

If your images are still in the original local project folder, point to that copy
without moving them:

```bash
python -m hsv_lbp_svm.svm_baseline --dataset /path/to/Dhan-Shomadhan --seeds 11 --validation-only
```

The script reads the audit and current-version pointer from this repository.
`--audit /path/to/preprocessing` can select another shared audit explicitly.
Caches and run files stay under the specified dataset directory.

## Run all five splits

Once the baseline settings are fixed, run:

```bash
python -m hsv_lbp_svm.svm_baseline
```

This uses seeds 11, 22, 33, 44 and 55. For each seed, it fits on training images,
chooses C and gamma using validation macro-F1, and evaluates the selected model
on the test images. The scaler is fitted only on training features, and the
selected model is not refitted on training plus validation data.

Existing run directories are protected from overwriting. For a separate
experiment, choose a new directory under the dataset, for example:

```bash
python -m hsv_lbp_svm.svm_baseline --output Dhan-Shomadhan/experiments/baseline_repeat
```

## Features and settings

The original group's feature design and search grid are retained:

- Resize each image to 256 x 256 and apply a 3 x 3 Gaussian blur.
- Extract an HSV histogram with 16 x 4 x 4 bins, normalised to sum to one.
- Extract uniform LBP histograms at (points=8, radius=1) and
  (points=16, radius=2), normalising each separately.
- Join the histograms into a 284-element feature vector.
- Apply `StandardScaler`, fitted separately for each training partition.
- Fit an RBF SVM with `class_weight='balanced'`.
- Search C in `{0.1, 1, 10, 100}` and gamma in `{'scale', 0.001, 0.01, 0.1}`.
  Select the highest validation macro-F1, retaining the first candidate in a tie.

Resizing and blurring are specific to this baseline. They are not changes to the
shared source images, and they do not imply that the ResNet methods should blur
their inputs. Feature extraction has no fitted parameters, so its cached values
can be reused between runs; scaling and classifier fitting remain separate for
each run. Source-image hashes and feature settings are checked before cache reuse.

## Files for the results member

Each `Dhan-Shomadhan/experiments/baseline/seed_<seed>/` contains:

| File | Contents |
|---|---|
| `run_config.json` | Shared split version, manifest checksums, features, selected parameters and package versions |
| `validation_history.csv` | Validation macro-F1 for all 16 C/gamma combinations |
| `selected_model.pkl` | Selected SVM, training-fitted scaler and disease mapping |
| `test_predictions.csv` | Image IDs, true/predicted disease IDs, background ID, method and seed |
| `metrics.json` | Accuracy, macro-F1, image counts and confusion matrices for field, white and combined (`all`) test images |

An all-five invocation also saves `summary_11_22_33_44_55.csv` with means and
sample standard deviations (`ddof=1`). A subset run records its actual run count;
standard deviation is left blank when there is only one run. Summary files cover
the seeds from that invocation, without mixing old runs with different settings.

Share the small configurations, validation histories, prediction files and
metrics. Model files and feature caches stay local. The output folders are ignored
by Git; see [group handover](../GROUP_HANDOVER.md) for adding selected result files.

## Generated images

The sample and candidate-pair images are shared preprocessing figures; they are
listed in the [figure catalogue](../Dhan-Shomadhan/preprocessing/figures/README.md).
The SVM currently writes numerical metrics and predictions, not feature-map or
confusion-matrix image files. Add any future SVM plots under
`Dhan-Shomadhan/experiments/baseline/figures/`, identifying the source seed and
background scenario. Do not put them with ResNet figures.

## Changes from the pasted script

The integration reads the three saved CSVs for each seed instead of the former
single-file schema. It uses the shared loader and checks manifest hashes and
group separation before fitting. The temporary random-split option was removed
because it could separate related images. The output now includes stable image
IDs and matches the group's prediction-file columns. Five-run summaries use
sample standard deviation, consistent with the handover.

The [SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
and [LBP documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern)
describe the library parameters used here.

## Integration checks

Checked on 26 October 2026: six SVM integration tests and the nine preprocessing
tests passed. The feature extractor matched the supplied script on three
comparison images. A seed-11 run completed on the real training/validation data:
the scaler was fitted to 777 training images, and 941 training/validation feature
vectors were cached. No real test predictions or test scores were produced in
that check. The 15 shared manifests were left unchanged.
