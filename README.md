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

For clean-checkout installation and exact rerun commands, see
[REPRODUCTION.md](REPRODUCTION.md). The official local experiment directories
used for the reported runs are `baseline`, `frozen_resnet18_8ep`,
`finetuned_resnet18_4ep` and `comparison_8ep_4ep` under the dataset's
`experiments/` folder. The shorter names under `results/` are compact committed
copies of those outputs.

## Shared split and input policy

All methods use `conservative_splits/bad2609eeda16df2` with seeds 11, 22, 33,
44 and 55. The targets are 70% training, 15% validation and 15% testing; group
constraints cause small count differences between runs. Related-image groups
never cross a partition within a run.

The reported Role 6 fine-tuned runs use the final shared 448×224 portrait
layout, ImageNet normalisation, and training-only augmentation. The frozen and
fine-tuned run configurations record the exact transform policy used per run.
The SVM uses its own
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
| Frozen ResNet-18 | White | 56.62 ± 7.12 | 50.58 ± 6.03 |
| Frozen ResNet-18 | Field | 49.20 ± 7.29 | 46.94 ± 9.50 |
| Frozen ResNet-18 | Combined | 54.36 ± 4.96 | 50.16 ± 5.16 |
| Fine-tuned ResNet-18 | White | 69.34 ± 4.69 | 64.18 ± 6.43 |
| Fine-tuned ResNet-18 | Field | 60.40 ± 6.54 | 57.92 ± 7.51 |
| Fine-tuned ResNet-18 | Combined | 66.62 ± 3.31 | 63.17 ± 5.12 |

The SVM used its full validation grid. The final Role 6 fine-tuning used five
shared seeds, validation-only candidate/epoch selection, and up to 20 additional
epochs; the selected epochs are recorded in each run configuration. The frozen
prerequisites were generated locally from the supplied frozen implementation
because trained Person 5 checkpoints were not included.

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
