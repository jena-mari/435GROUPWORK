# Preprocessing figures

These images describe the dataset and its audit. They are shared across the
methods and do not show classifier accuracy or CNN features.

| Files | Purpose | Suggested report section |
|---|---|---|
| `samples_raw.png` | One example per disease/background, with original aspect ratios | Problem Description; reference in Dataset Preparation |
| `samples_eligible.png` | Examples from the retained inventory | Dataset Preparation; currently identical to the raw examples because all images were retained |
| `pairs_strong_00.jpg` to `pairs_strong_03.jpg` | The 32 candidate pairs with the smallest summed hash distances | Dataset Preparation and related-image grouping |
| `contact_<disease>_<page>.jpg` | Class contact sheets for broader visual inspection | Audit review or selected examples in the appendix |

The two images previously placed at the repository root (`samples_raw.png` and
`pairs_strong_00.jpeg`) were exact copies of the canonical sample and first pair
sheet. Their canonical copies are stored here, without changing the image bytes.

Candidate-pair sheet caption:

> **Figure X.** Examples flagged by perceptual-hash screening as potentially
> related images. Some pairs show repeated views, while others share similar
> leaf outlines or backgrounds. Similarity alone does not confirm shared-leaf
> identity; unresolved candidates were conservatively kept within the same partition.

Split-count table caption (the source CSV remains in the split version directory):

> **Table X.** Achieved image counts for five group-preserving repeated holdout
> splits targeting 70% training, 15% validation and 15% testing. Counts vary
> slightly because related-image groups remain indivisible. Each partition
> contains all five disease classes under both background conditions.

Rebuild these figures from the saved inventory, review file and thumbnails:

```bash
python data_preprocessing/make_figures.py
```

Run from the repository root with the preprocessing dependencies installed. The
generator does not change labels, review decisions or train/validation/test
membership. It replaces the generated image files and leaves the audit CSVs unchanged.
The review records remain the source for which relationships are visually
supported and which are unresolved.
