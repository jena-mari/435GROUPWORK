# Person 6 — Fine-tuned ResNet-18 experiments

## Completion and contribution

This submission implements the partly fine-tuned ResNet-18 workflow, selects settings using validation data, and provides test predictions, logs, checkpoints, figures, and interpretation for all five shared seeds. Person 5's supplied ZIP contained training code but no trained checkpoints or results. The prerequisite frozen checkpoints were therefore generated locally using the supplied `train_frozen.py` unchanged. These prerequisite results should be coordinated with Person 5; they are not claimed to have been supplied as completed runs.

## Method and implementation

Each run starts from the best frozen checkpoint for its matching seed. ImageNet-pretrained ResNet-18 uses a five-class linear classifier. Only `layer4` and `fc` are optimised; the stem and layers 1–3 remain fixed. All BatchNorm running statistics stay fixed, including during training; affine parameters in `layer4` may learn. The shared 448 × 224 portrait input, ImageNet normalisation, and training-only augmentation are reused unchanged. Background IDs are retained as evaluation metadata and are not model inputs or targets.

Adam uses batch size 32 for at most 20 additional epochs per candidate. A predeclared validation search compares (`layer4`, `fc`) learning rates: candidate 0 uses `(1e-5, 1e-4)` and candidate 1 uses `(3e-5, 3e-4)`. Each candidate restarts from the same frozen checkpoint and model/augmentation seed. Highest validation macro-F1 selects the epoch and candidate; exact ties use lower validation loss, then the earlier candidate. Training stops after five consecutive epochs without improvement. All five fine-tuning selections finish before any fine-tuned test evaluation. No test metric is used to choose settings.

The saved split version is `conservative_splits/bad260e9eda16df2`. Seeds 11, 22, 33, 44, and 55 are repeated random holdouts, not five-fold cross-validation. Every run has 777 training images; validation/test counts differ slightly. Configuration files record source-checkpoint hashes, manifest hashes, code hashes, trainable stages, learning rates, selected epoch, transforms, device, and package versions.

## Test results

| Seed | Images | Accuracy | Macro-F1 | Candidate | Selected epoch |
|---|---:|---:|---:|---:|---:|
| 11 | 165 | 0.7091 | 0.6977 | 1 | 17 |
| 22 | 164 | 0.6280 | 0.5610 | 1 | 11 |
| 33 | 165 | 0.6424 | 0.6231 | 1 | 6 |
| 44 | 164 | 0.6890 | 0.6605 | 1 | 11 |
| 55 | 166 | 0.6627 | 0.6161 | 1 | 8 |

| Background subset | Accuracy, mean ± sample SD | Macro-F1, mean ± sample SD |
|---|---:|---:|
| field | 0.6040 ± 0.0654 | 0.5792 ± 0.0751 |
| white | 0.6934 ± 0.0469 | 0.6418 ± 0.0643 |
| all | 0.6662 ± 0.0331 | 0.6317 ± 0.0512 |

## Interpretation

Across the five holdouts, combined test accuracy is 0.6662 ± 0.0331 and macro-F1 is 0.6317 ± 0.0512. The mean paired macro-F1 difference from the locally generated frozen prerequisites is +0.1301 (+13.01 percentage points). Fine-tuning improved mean macro-F1 in these runs.

Mean field-background macro-F1 is 0.5792; mean white-background macro-F1 is 0.6418. The strongest disease by mean combined per-class F1 is Sheath Blight (0.8464), and the weakest is Leaf Scald (0.4844). The largest off-diagonal confusion pooled over the five test runs is Leaf Scald predicted as Rice Blast (31 prediction events). Those events may include the same image in more than one holdout.

These are descriptive results, not evidence of statistical significance. Holdouts overlap, so their scores are not independent replicates. The same disease labels and exact test images must be used for comparisons with the SVM baseline and Person 5's final results. Combined macro-F1 is computed from combined predictions, rather than averaging field and white macro-F1.

Field/white performance differences may reflect both image difficulty and different class/group composition; they do not establish a causal background effect. The conservative grouping policy assigns a 461-image white-background group to training in every run, limiting test coverage. Label agreement with metadata does not independently verify biological diagnoses. Learning curves can reveal divergence between training and validation, while confusion matrices and per-class F1 identify the particular diseases that remain difficult. The failure CSV lists all misclassified test images by seed for Person 4's case analysis.

## Dataset attribution

Md Fahad Hossain, Sheikh Abujar, Sheak Rashed Haider Noori and Syed Akhter Hossain (2021). Dhan-Shomadhan: A Dataset of Rice Leaf Disease Classification for Bangladeshi Local Rice. Mendeley Data, V1. DOI: `10.17632/znsxdctwtt.1`. The project's dataset documentation identifies the licence as Creative Commons Attribution 4.0 International. Original dataset images are not redistributed in this package.

Architecture reference: Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun (2016). Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition. DOI: `10.1109/CVPR.2016.90`. [Official publication record](https://www.microsoft.com/en-us/research/publication/deep-residual-learning-for-image-recognition/).

## Figures and files

`figures/learning_curves_seed_*.png` shows both candidates and the selected validation checkpoint. `figures/confusion_matrices_seed_*.png` shows field, white, and combined test confusion matrices. `seed_*/` contains the selected configuration, candidate histories and validation predictions, test predictions and metrics. `per_class_f1.csv`, `paired_difference_from_frozen.csv`, and `misclassified_test_images.csv` support the group's evaluation section.

## Verification

All 1,106 original images, source metadata, and 15 manifests passed the audited checksum verifier. Submission generation checks each selected checkpoint hash, manifest hashes, code hash, test image IDs, true labels, background IDs, probability sums/ranges, and prediction classes. Metrics are independently recomputed from exported test CSVs and must match the saved metrics exactly. Automated tests verify that only `layer4`/`fc` change, BatchNorm statistics remain fixed, mismatched frozen sources are rejected, and validation selection never predicts the test partition.
