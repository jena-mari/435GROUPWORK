# CNN implementation evidence

These are small shared records for implementation review. Final model results
belong in the dataset's `experiments/<method>/` directories.

| File | What it records |
|---|---|
| `common_seed11_mps.json` | Original small two-stage shared-component check |
| `frozen_integration_seed11_mps.json` | Integrated frozen experiment check with no test evaluation |
| `shared_components_seed11_mps.json` | Consolidated shared-package check: all saved splits plus small training/validation runs for both stages |
| `transform_preview_seed11.png` | Original training images, evaluation preprocessing and three augmentation examples |
| `transform_preview_seed11.metadata.json` | Selected source IDs, seed, manifests, policy and code hashes for that preview |

The preview supports the Common Preprocessing section and helps group members
inspect how padding and rotations affect narrow leaves. It shows transformed
inputs, not classifications. Its source images are attributed to Hossain et al.,
Dhan-Shomadhan V1, DOI `10.17632/znsxdctwtt.1`, CC BY 4.0; see the
[dataset attribution](../../README.md#dataset-attribution). Display panels are
resized and augmented, with fixed ImageNet normalisation inverted for viewing.

The checks use small training/validation subsets. Their losses and validation
scores are diagnostic; they do not establish five-run accuracy or macro-F1.
Large checkpoints and local environments are excluded from this folder.
