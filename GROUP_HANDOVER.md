# Dataset handover

The dataset preparation, SVM baseline and frozen ResNet-18 experiment are ready
to run. Shared CNN components and the evaluation package are available; the
fine-tuned experiment driver still needs to be supplied. Use the saved splits so
that each method is evaluated on the same images. Final model results have not yet been added to the
repository.

## Folder responsibilities

- **Shared preparation:** `data_preprocessing/` contains the audit, loader,
  split generator and figure generator. Shared data and manifests remain under
  `Dhan-Shomadhan/preprocessing/`.
- **HSV–LBP–SVM:** `hsv_lbp_svm/` contains the implementation, requirements,
  tests and method guide. Local results stay under `experiments/baseline/` inside
  the dataset directory.
- **Shared CNN components (Role 3):** `resnet18/common.py`, `check_cnn.py` and
  the shared tests provide the agreed inputs, pretrained loading, checkpoint and
  prediction contracts. Use the [component handover](resnet18/SHARED_COMPONENTS_INTEGRATION.md)
  for the supplied-package mapping and transformation preview.
- **ResNet-18:** `resnet18/` contains shared components, unit tests and a small
  implementation-check runner and the integrated Role 5 frozen experiment.
  Role 6 still needs its full experiment driver. Final CNN test results are pending.
- **Evaluation:** `evaluation/` validates saved predictions against all five test
  manifests and generates comparative tables and matrices. Supplied thumbnail-SVM
  results stay separately labelled under `evaluation/reference/`.
- **Inspection images:** `Dhan-Shomadhan/preprocessing/figures/` contains the
  sample figure, candidate pairs and contact sheets. These belong to the shared
  data analysis. Future model figures belong with their model's experiment.

Read the [dataset map](Dhan-Shomadhan/README.md) for full output paths.

## First steps for every member

1. Follow the [README](README.md) to clone the repository, download version 1 of
   Dhan-Shomadhan, and install the preprocessing dependencies with Python 3.12.
2. From the repository root, run:

   ```bash
   python data_preprocessing/verify_dataset.py
   python -m unittest discover -s data_preprocessing -p 'test_prepare_dataset.py' -v
   ```

   The first command should report that all **1,106 images**, the metadata CSV
   and **15 manifests** match. The second should pass nine tests. Stop and check
   the extraction path if images are missing; do not rename folders or edit the
   manifests to make the check pass.
3. Read the current split version from
   [`CURRENT_SPLIT_VERSION.txt`](Dhan-Shomadhan/preprocessing/CURRENT_SPLIT_VERSION.txt).
   The current value is `conservative_splits/bad260e9eda16df2`.
4. Start with seed 11 to check that your feature extraction or training code runs.
   Once it works, repeat it for **11, 22, 33, 44 and 55**. Each run needs its own
   fitted preprocessing, model and checkpoint.

Before starting a new session, use `git pull --ff-only` to pick up shared code and
documentation updates. If you have local changes, commit them or coordinate with
the group before pulling. Keep the recorded split version fixed for the comparison.

The raw images are downloaded separately. The images under `preprocessing/`
are small inspection previews and figures; train on the original images loaded
through the manifests.

## What has been prepared

| Item | Location | How it is used |
|---|---|---|
| Image inventory | `Dhan-Shomadhan/preprocessing/inventory.csv` | Original paths, supplied labels, dimensions, hashes and readability checks |
| Disease/background counts | `Dhan-Shomadhan/preprocessing/counts_raw.csv` | Dataset description and class/background imbalance |
| Inspection figure | `Dhan-Shomadhan/preprocessing/figures/samples_raw.png` | One example of each disease in each background for the report |
| Similarity review | `Dhan-Shomadhan/preprocessing/pair_review.csv` | Candidate relationships and recorded visual decisions |
| Model inputs | Current version's `seed_*/train.csv`, `validation.csv`, `test.csv` | Fixed image membership for each run |
| Label mapping | Current version's `label_mapping.json` | The same disease IDs and output order for every method |
| Audit trail | Current version's `provenance.json` and `effective_pair_decisions.csv` | Split settings, checksums and grouping decisions |
| Report draft | `Dhan-Shomadhan/preprocessing/report_sections.txt` | Problem description, image challenges and preparation method |

Open `dataset_report.html`, `review.html` or `gallery_*.html` in a local browser
to inspect the formatted report and previews. The HTML files are stored in
`Dhan-Shomadhan/preprocessing/`.

All 1,106 original images are retained. The audit found no unreadable files,
byte-identical copies, or disagreements between the supplied metadata and the
folder labels after spelling variants were mapped. These checks do not verify
the biological diagnosis of each leaf.

## Loading the data

Run code from the repository root. This example loads the three partitions for
one run:

```python
from pathlib import Path
from data_preprocessing.prepare_dataset import RiceDataset

root = Path('Dhan-Shomadhan')
audit = root / 'preprocessing'
version = audit / (audit / 'CURRENT_SPLIT_VERSION.txt').read_text().strip()
seed = 11

datasets = {
    part: RiceDataset(root, version / f'seed_{seed}' / f'{part}.csv')
    for part in ('train', 'validation', 'test')
}

print({part: len(dataset) for part, dataset in datasets.items()})
# Seed 11: {'train': 777, 'validation': 164, 'test': 165}

sample = datasets['train'][0]
print(sample['image'].mode, sample['disease_id'], sample['background_id'])
```

Each sample contains:

| Field | Meaning |
|---|---|
| `image` | RGB PIL image, or the result of the supplied `transform=` callable |
| `disease_id` | Classification target: Brown Spot=0, Leaf Scald=1, Rice Blast=2, Rice Tungro=3, Sheath Blight=4 |
| `background_id` | Evaluation metadata: field=0, white=1 |
| `image_id` | Stable identifier used to match predictions to the manifest |

For handcrafted features, convert the PIL image with `np.asarray(image)` after
importing NumPy. For a PyTorch batch, pass a transform that produces a tensor of
a fixed size. Without that transform, the loader returns PIL images, which the
standard batch collation does not handle directly. Use random augmentation only
for the training dataset and a deterministic transform for validation and test.

The dataset may stay in another local directory. Pass that directory as
`RiceDataset`'s first argument and keep the manifest path in this repository.
To verify it against the repository's audit, run:

```bash
python data_preprocessing/verify_dataset.py --root /path/to/Dhan-Shomadhan --audit Dhan-Shomadhan/preprocessing
```

## Work for each method

| Task | Implementation to complete | What to return |
|---|---|---|
| Traditional baseline | Run `python -m hsv_lbp_svm.svm_baseline` using [the baseline guide](hsv_lbp_svm/README.md). It extracts HSV/LBP features and selects an RBF SVM on validation macro-F1. The scaler is fitted on training features only. | Run configurations, validation histories, test predictions and metrics for all five seeds |
| Frozen ResNet-18 | Run `python -m resnet18.train_frozen` using the [frozen guide](resnet18/ROLE5_FROZEN_EXPERIMENTS.md). The integrated loop freezes the backbone and its BatchNorm statistics and selects the five-class head by validation macro-F1. | Weight source, transforms, selected checkpoint/epoch, validation history and test predictions for each seed |
| Fine-tuned ResNet-18 | Start from the best frozen checkpoint for the **same seed**. Train `layer4` and the head while earlier stages remain fixed. Keep the agreed BatchNorm running-statistics policy. | Frozen checkpoint used, trainable layers, learning rates, selected checkpoint/epoch and test predictions for each seed |
| Results and report | Use [the evaluation package](evaluation/README.md) to validate all five runs against the shared manifests and calculate metrics for the three backgrounds. Inspect original images for error analysis. | Per-run scores, means and standard deviations, confusion matrices, example errors and limitations |

The full-run ResNet settings from the project plan are below. Common components
have passed unit and small-subset checks, but the full five-run experiments have
not been completed. Record actual settings and any changes in each run configuration.

| Setting | Frozen model | Fine-tuned model |
|---|---|---|
| Starting weights | ImageNet-pretrained ResNet-18 | Best frozen-model checkpoint from the same split |
| Trainable parameters | Five-class head | `layer4` and five-class head |
| Optimiser | Adam | Adam |
| Learning rate | Head: `1e-3` | `layer4`: `1e-5`; head: `1e-4` |
| Batch size | 32 | 32 |
| Maximum epochs | 30 | 20 additional epochs |
| Selection metric | Validation macro-F1 | Validation macro-F1 |
| Early stopping | Five consecutive epochs without improvement | Same rule, with its own validation selection |

Both variants use the approved input-size, normalisation and augmentation
policy described below. Changes require a separate documented experiment. Calling
`model.train()` also switches BatchNorm modules into training mode, so reapply the chosen evaluation-mode
settings after that call. Freezing parameter gradients alone does not freeze
BatchNorm running statistics.

The relevant PyTorch references are
[data loading and collation](https://docs.pytorch.org/docs/2.14/data.html),
[training/evaluation mode](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)
and [BatchNorm running statistics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html).

The [ResNet-18 guide](resnet18/README.md) and
[role validation checklist](resnet18/ROLE_VALIDATION.md) describe integration.
Both variants now share 224 x 224 aspect-preserving resize/padding and ImageNet
normalisation. Training additionally uses horizontal flips (probability 0.5) and
expanded rotations up to 10 degrees, followed by resize/padding again. Validation
and test have no random augmentation. The full policy is checked in checkpoints.
The frozen experiment uses the common optimiser/checkpoint helpers and loads
the selected same-context checkpoint before exporting predictions. Role 6 must
use the same helpers and start from that seed's `best.pt`, with a new optimiser
and independent validation selection. Diagnostic checkpoints are rejected.

## Frozen experiment and evaluation commands

Use `.venv-cnn` from the repository root. Substitute the original image directory
for `/path/to/Dhan-Shomadhan`:

```bash
python -m resnet18.train_frozen --dataset /path/to/Dhan-Shomadhan --validation-only
python -m resnet18.train_frozen --dataset /path/to/Dhan-Shomadhan --evaluate-only
python -m pip install -r evaluation/requirements.txt
python -m evaluation.role4_evaluation \
  --svm /path/to/Dhan-Shomadhan/experiments/baseline \
  --frozen /path/to/Dhan-Shomadhan/experiments/frozen_resnet18 \
  --finetuned /path/to/Dhan-Shomadhan/experiments/finetuned_resnet18 \
  --out /path/to/Dhan-Shomadhan/experiments/comparison
```

The first command trains all five frozen runs without test evaluation. The
second exports test predictions from those selected checkpoints. Run it only
once the training procedure is fixed. The comparison command requires all five
runs of each supplied method. Omit a pending method until it is ready; the
output will record that the three-method comparison is incomplete.

Role 4's supplied thumbnail results and PDFs are preserved under
`evaluation/reference/role4_thumbnail_svm/`. They used thumbnails and fixed SVM
settings, so they cannot stand in for the final baseline. The integrated evaluator
has a separate `--thumbnail-reference` mode to check them.

## Rules for the comparison

Use the saved membership for every method. The achieved image counts are:

| Seed | Train | Validation | Test |
|---|---:|---:|---:|
| 11 | 777 | 164 | 165 |
| 22 | 777 | 165 | 164 |
| 33 | 777 | 164 | 165 |
| 44 | 777 | 165 | 164 |
| 55 | 777 | 163 | 166 |

These are five repeated random holdouts targeting 70/15/15. Group sizes prevent
exact percentages, and images may change partitions between runs. They are not
five-fold cross-validation.

Fit each method on training data and choose its configuration and checkpoint
using validation data. Keep test data out of those choices. After the procedure
is fixed, evaluate each selected checkpoint on its test set. Background labels,
image IDs, file paths and group IDs must not become classifier input features.

Set and record the model's random seed as well as the split seed. Record package
versions, device and any determinism settings. Fixed manifests reproduce image
membership; they do not by themselves make training deterministic.

## Saving results for the group

Keep local run files under
`Dhan-Shomadhan/experiments/<method>/seed_<seed>/`, using method names
`baseline`, `frozen_resnet18` and `finetuned_resnet18`.

For each run, save:

- `run_config.json`: method, seed, manifest version, code commit, transforms,
  feature/model settings, package versions, device and checkpoint-selection rule.
- `validation_history.csv`: epochs and validation scores for the neural models;
  for the baseline, record the candidate settings and validation scores tested.
- `test_predictions.csv`: one row per test image, using the columns below.
- `metrics.json`: per-run metrics for field, white and combined test images.
- The selected model or fitted pipeline locally, with its location recorded in
  `run_config.json`.

Prediction-file columns:

```text
image_id,seed,method,disease_id,predicted_disease_id,background_id
```

Read the true disease and background IDs from the manifest. Match results by
`image_id`, since a shuffled loader changes row order. Before scoring, check for
missing IDs, repeated IDs and predictions for images outside that run's test
manifest. If probabilities are saved, use columns `prob_0` to `prob_4` in the
fixed disease-ID order.

For each seed, score the field subset (`background_id == 0`), white subset
(`background_id == 1`) and combined test set. Report accuracy and macro-F1, plus
per-class scores and confusion matrices. Compute each subset's metrics from its
own predictions; combined macro-F1 is not an average of the two background
macro-F1 scores. Then report the mean and sample standard deviation (`ddof=1`)
of the five per-run scores for each method and scenario. Keep the per-run scores
available so the methods can also be compared seed by seed.

Place model-generated figures in `experiments/<method>/figures/`, with seed and
background scenario in the filename. Keep shared inspection figures in
`preprocessing/figures/`.

The `experiments/` directory is ignored by Git. Share the small configuration,
prediction and metric files with explicit paths when they are ready, for example:

```bash
git add -f Dhan-Shomadhan/experiments/baseline/seed_11/run_config.json Dhan-Shomadhan/experiments/baseline/seed_11/test_predictions.csv Dhan-Shomadhan/experiments/baseline/seed_11/metrics.json
```

Keep large model checkpoints and temporary feature arrays out of the commit.
The group member combining results needs the small files for all five seeds.

## How this preparation supports further work

The fixed image IDs, labels and partitions let us change one method at a time
and compare its predictions with the others on the same test images. This is the
basis for testing whether pretrained features improve on handcrafted features,
and whether training `layer4` adds a further improvement.

Background metadata lets us check whether an overall score hides weaker field
performance. There are 769 white-background images and 337 field images, so the
combined result alone does not describe both settings equally. The five splits
also show how much each method's score changes with the allocation of images.
Their samples overlap, so the runs should not be described as five independent
datasets.

Useful follow-up experiments include comparing resize/crop policies, testing
augmentation choices, reviewing the similarity groups, and testing on a separate
rice-image collection. Define each follow-up before examining its test results.
Use validation data to choose preprocessing settings. A field-only versus
white-only training experiment needs its own documented, group-separated split
design; the present mixed-background holdouts do not establish transfer to a new
acquisition setting.

The current grouping is the main issue to revisit. It keeps unresolved candidate
images together to reduce leakage risk, but connects 461 white-background images
into one group. That group is in training in every run, so those images never
contribute to the test scores. Visual review could separate unrelated leaves and
improve test coverage. Any revised grouping needs a new version and a rerun of
all compared methods. Keep the current version for the existing comparison; do
not change it partway through one member's experiments.

## Finishing the report

Use the audited counts and Figure 1 for the Problem Description, and explain how
clutter, overlapping leaves, lighting, small lesions and class imbalance affect
the methods. The preparation draft describes the implemented pipeline. Check
its wording against the final training code before submission.

The report should state that perceptual candidates were conservatively grouped,
that full pairwise review is unfinished, and that plant/session identifiers were
not supplied. The audit does not demonstrate field deployment performance or
independent plant-level generalisation. Healthy leaves and diseases outside the
five supplied classes were not evaluated.

Verify each literature citation against the original paper, including its venue,
year and the claim it supports. The current preparation does not validate the
Ahmad or Bailey references in the research draft. Follow the assignment's source
and submission requirements when assembling the final report and code archive.
