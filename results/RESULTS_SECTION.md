# V. Experimental Results and Analysis

## V.1. Experimental settings

The comparison used five versioned, group-preserving holdout splits with seeds
11, 22, 33, 44 and 55. Each split targeted 70% training, 15% validation and
15% testing. The achieved counts were 777 training images, 163–165 validation
images and 164–166 test images. Related-image groups stayed within one
partition. Validation macro-F1 selected SVM parameters and neural-network
checkpoints. Test images were not used for either decision.

The HSV–LBP–SVM resized images to 256×256, applied a 3×3 Gaussian filter and
used 284 HSV/LBP features. It searched C values 0.1, 1, 10 and 100 and gamma
values `scale`, 0.001, 0.01 and 0.1. A StandardScaler was fitted separately on
each training partition. The frozen ResNet-18 replaced its classifier with a
512-to-5 linear head and trained it with Adam at learning rate 10⁻³. The
fine-tuned model started from the selected frozen checkpoint for the same seed
and trained `layer4` and the head with learning rates 10⁻⁵ and 10⁻⁴. Both CNNs
used batch size 32, ImageNet normalisation, 224×224 aspect-preserving padding,
training-only horizontal flips and rotations up to ±10°, with fixed BatchNorm
running statistics.

The SVM used its complete validation grid. The frozen runs used an eight-epoch
maximum and the fine-tuned runs used four additional epochs. These are within
the planned maxima of 30 and 20 epochs and are recorded in the run files.

## V.2. Results

Table 1 reports mean ± sample standard deviation over the five runs. Scores are
percentages. Combined scores are calculated directly from all test predictions.

| Method | Background | Accuracy | Macro-F1 |
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

*Table 1. Five-run test performance on the shared holdouts.*

The SVM achieved the highest combined macro-F1 at 52.17%. Fine-tuning improved
the combined macro-F1 over the frozen model from 43.08% to 49.35%, although it
remained below the SVM result. Fine-tuning also reduced run-to-run variation:
the combined macro-F1 standard deviation decreased from 7.41 to 2.20 points.

## V.3. Discussion

The results partly support the expectation that adapting higher-level features
would help. Fine-tuning improved the frozen representation, especially on field
images, but did not surpass the handcrafted SVM under the recorded training
caps. The frozen model was less stable across splits, as shown by its larger
standard deviations.

The SVM was strongest on the combined test set and was relatively balanced
between the two backgrounds. Its average class-level F1 was highest for Sheath
Blight and Rice Tungro, while Brown Spot was more difficult, particularly in the
white-background subset. The fine-tuned model also performed best on Sheath
Blight and Rice Blast, but Brown Spot remained its weakest class. Use
`results/comparison/prediction_review.csv` and the original images to select
specific examples for the final report.

Field images were more difficult for both ResNet variants: the frozen macro-F1
dropped from 43.13% on white images to 38.75% on field images, and the
fine-tuned score dropped from 50.36% to 43.68%. The SVM showed a smaller
background gap. This is consistent with field clutter and lighting making lesion
appearance less consistent, but the experiment does not establish background as
the sole cause.

Possible method improvements include higher-resolution or leaf-focused inputs
for small lesions, a controlled lighting augmentation study, and longer or
better-tuned fine-tuning selected only with validation data. Each change should
use a new documented configuration. The results are limited to the supplied
images and conservative grouping design; the repeated holdouts overlap and do
not measure generalisation to unseen farms, plants or cameras.
