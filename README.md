# Rice leaf disease classification

This repository contains the shared data preparation, an HSV–LBP–SVM baseline,
two ResNet-18 experiments and the evaluation files for the CSCI435 project.
The source images are downloaded separately because the archive is about 1.8 GB.

## Start here

1. Clone the repository and download version 1 of Dhan-Shomadhan from the
   [Mendeley Data record](https://data.mendeley.com/datasets/znsxdctwtt/1).
2. Extract the archive so the original folders remain under `Dhan-Shomadhan/`.
3. Run the audit check:

   ```bash
   python data_preprocessing/verify_dataset.py \
     --root /path/to/Dhan-Shomadhan \
     --audit Dhan-Shomadhan/preprocessing
   ```

The audit expects 1,106 readable images, the supplied metadata file and the 15
saved manifests. Do not rename the source folders; two of them contain trailing
spaces in the original dataset.

## Repository map

| Folder | Purpose |
|---|---|
| `data_preprocessing/` | Audit, loader, duplicate review and shared splits |
| `hsv_lbp_svm/` | Full-resolution HSV–LBP–SVM baseline |
| `resnet18/` | Shared CNN code, frozen and fine-tuned drivers, checks |
| `evaluation/` | Manifest checks and cross-method comparison |
| `results/` | Compact outputs from the completed five-run experiments |
| `Dhan-Shomadhan/preprocessing/` | Versioned audit records and manifests |

The raw images, model checkpoints, feature caches and local environments are
ignored by Git. The committed prediction files in `results/` are enough to
reproduce the reported tables and inspect errors by image ID.

## Shared split and input policy

All methods use `conservative_splits/bad2609eeda16df2` with seeds 11, 22, 33,
44 and 55. The targets are 70% training, 15% validation and 15% testing; group
constraints cause small count differences between runs. Related-image groups
never cross a partition within a run.

The CNNs use 224×224 aspect-preserving padding, ImageNet normalisation, and
training-only horizontal flips plus rotations up to ±10°. The SVM uses its own
256×256 resize, Gaussian blur and HSV/LBP features. Background is evaluation
metadata, not a model input.

## Completed experiments

The results below are from the full 1,106-image dataset. Values are mean ± sample
standard deviation over the five shared holdouts. Accuracy and macro-F1 are
percentages.

| Method | Test subset | Accuracy | Macro-F1 |
|---|---|---:|---:|
| HSV–LBP–SVM | White | 54.88 ± 3.59 | 51.24 ± 1.72 |
| HSV–LBP–SVM | Field | 51.60 ± 4.56 | 51.33 ± 5.37 |
| HSV–LBP–SVM | Combined | 53.88 ± 3.42 | 52.17 ± 2.59 |
| Frozen ResNet-18 | White | 50.52 ± 8.04 | 43.13 ± 6.24 |
| Frozen ResNet-18 | Field | 41.20 ± 9.23 | 38.75 ± 9.00 |
| Frozen ResNet-18 | Combined | 47.69 ± 8.11 | 43.08 ± 7.41 |
| Fine-tuned ResNet-18 | White | 54.70 ± 1.42 | 50.36 ± 3.34 |
| Fine-tuned ResNet-18 | Field | 46.00 ± 5.83 | 43.68 ± 5.38 |
| Fine-tuned ResNet-18 | Combined | 52.06 ± 2.31 | 49.35 ± 2.20 |

The SVM used its full validation grid. To keep the CNN runs practical on the
available machine, the frozen runs used an 8-epoch cap and the fine-tuned runs
used a 4-epoch additional-training cap. Both caps are within the planned limits
of 30 and 20 epochs, respectively, and are recorded in each run configuration.

Open `results/comparison/five_run_summary.csv` for the unrounded values. The
three pooled confusion matrices are in `results/comparison/figures/`; pooled
counts include repeated test appearances because the holdouts overlap.

## Run the code

Use Python 3.12 and separate environments for the SVM and CNNs:

```bash
python -m unittest discover -s data_preprocessing -p 'test_*.py' -v
python -m unittest discover -s hsv_lbp_svm/tests -p 'test_*.py' -v
python -m unittest discover -s resnet18/tests -p 'test_*.py' -v
python -m unittest discover -s evaluation/tests -v
```

The SVM command is documented in `hsv_lbp_svm/README.md`. For the CNNs, start
with `python -m resnet18.check_cnn check --dataset /path/to/Dhan-Shomadhan`, then
read `resnet18/ROLE5_FROZEN_EXPERIMENTS.md` and
`resnet18/SHARED_COMPONENTS_INTEGRATION.md`.

To recompute the comparison from saved predictions:

```bash
python -m evaluation.role4_evaluation \
  --svm results/baseline \
  --frozen results/frozen_resnet18 \
  --finetuned results/finetuned_resnet18 \
  --out /tmp/rice-comparison
```

## Attribution and limits

Dhan-Shomadhan is attributed to Hossain et al. (2021), Mendeley Data V1,
DOI `10.17632/znsxdctwtt.1`, CC BY 4.0. The audit checks file readability,
labels and duplicates; it does not establish biological diagnosis or independent
plant-level sampling. The five runs are repeated holdouts, not five-fold
cross-validation. The results describe this dataset and split design, rather
than deployment performance on new farms or cameras.
