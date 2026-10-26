DHAN-SHOMADHAN: SHARED DATA PREPARATION

Files: prepare_dataset.py, requirements.txt, test_prepare_dataset.py.
Use Python 3.12. Install with:
    python -m pip install -r data_preprocessing/requirements.txt
Pillow is an explicit supporting dependency for image decoding, EXIF orientation,
RGB conversion and figures. NumPy implements hash calculations and allocation.
The loader does not import PyTorch; it implements the map-style Dataset protocol.
Use your model environment's PyTorch/torchvision for training.

From the project directory, with scripts in data_preprocessing/:
    python data_preprocessing/prepare_dataset.py audit
    # Inspect Dhan-Shomadhan/preprocessing/review.html and gallery_*.html.
    # Edit pair_review.csv: related / distinct / conservative, plus a reason.
    # Use image_decisions.csv for documented corrections, exclusions, or groups
    # missed by hash screening. Repeated manual_group values join those images.
    python data_preprocessing/prepare_dataset.py build --gallery-reviewed

Selected policy for this project: conservatively keep unresolved candidates
together, without pretending that all candidate pairs have been visually reviewed:
    python data_preprocessing/prepare_dataset.py build --conservative-pending
This saves conservative_splits/<version>/seed_*/{train,validation,test}.csv.
The original pending reviews remain editable; effective_pair_decisions.csv records
exactly which unresolved edges were grouped in each saved version.
This is the chosen evaluation design, not proof that every group is one leaf.
It protects against leakage through screened candidate edges, at the cost of
possible over-grouping and reduced test representativeness. Full visual review
can improve the design in future. Related images missed by screening remain a risk.

--root accepts another dataset directory. By default all generated artefacts are
under ROOT/preprocessing. --output permits temporary audit staging elsewhere.
Audit refuses to overwrite a previous audit; review decisions must be preserved.
Build saves a version identified by the inventory, reviews, code and parameters.
Each version contains seed_11, seed_22, seed_33, seed_44 and seed_55 directories,
each with train.csv, validation.csv, test.csv and achieved stratum counts.
Provenance records file hashes, packages, target fractions and the allocation rule.
Keep one version shared across all three methods. Do not rerun splits per model.

AUDIT
All original paths and original folder labels are retained. Whitespace-normalised
aliases map spelling variants to Brown Spot, Leaf Scald, Rice Blast, Rice Tungro
and Sheath Blight, with IDs 0..4. Field=0 and white=1. The supplied metadata CSV
is checked against folder labels and image filenames; this checks consistency,
not biological correctness. The two sources may share the same original errors.

Readability uses Pillow verify(), full decoding and EXIF-aware RGB conversion.
Unreadable images are excluded with a recorded reason. Full-file SHA-256 detects
byte-identical copies. One lexicographically first copy is retained only when its
resolved disease/background labels agree; conflicts block building until resolved.
No original image is changed or deleted.

Near-duplicate screening: horizontal 64-bit dHash at Hamming distance <=6 OR
64-bit pHash at distance <=8. pHash uses a 32x32 grayscale orthonormal DCT-II,
the upper-left 8x8 block and the median of its 63 AC coefficients. The DC bit is
retained. These are fixed exploratory screening thresholds, not learned or
validated same-leaf detectors. Crops, rotations and background changes can cause
misses. Similar shapes can cause false candidates. Inspect whole-class galleries
for missed repeated leaves and record groups before generating splits.

Pair decisions:
  related: visual evidence supports the same leaf or another related acquisition.
  distinct: visual inspection supports separate leaves/images; document why.
  conservative: uncertain; retain both and keep together to reduce leakage risk.
Unreviewed candidates block final split generation. Pair grouping is transitive;
exact duplicates are joined before removing redundant copies so no relationships
are lost. Without plant/session identifiers, independent biological sampling
cannot be guaranteed. Do not claim that perceptual hashing proves independence.

Whole-image luminance mean/SD and 256x256 Laplacian variance are descriptive
proxies. They are affected by background, symptom texture and resizing. They do
not establish disease severity, camera quality or grounds for automatic exclusion.

SPLITS
These are five repeated random holdouts, not five-fold cross-validation.
Within each seed, indivisible related-image groups are allocated to 70/15/15
targets using 200 seeded greedy starts. The objective is the sum over all 30
partition/stratum cells of squared count deviation divided by target count.
The best valid start is retained. This heuristic is deterministic in the recorded
environment, but does not guarantee a globally optimal allocation. Exact target
percentages may be infeasible with groups; report achieved counts and fractions.
All ten disease/background strata must appear in each partition, and no image,
group or exact hash may cross partitions. Infeasible allocation fails explicitly.
Images can change partitions across runs. Report means and sample standard
deviations across runs, while recognising that holdout runs overlap.

LOADER EXAMPLE
    from pathlib import Path
    from data_preprocessing.prepare_dataset import RiceDataset
    root = Path('Dhan-Shomadhan')
    relative_version = (root/'preprocessing/CURRENT_SPLIT_VERSION.txt').read_text().strip()
    version = root / 'preprocessing' / relative_version
    training = RiceDataset(root, version/'seed_11/train.csv', transform=train_transform)
    validation = RiceDataset(root, version/'seed_11/validation.csv', transform=eval_transform)
    testing = RiceDataset(root, version/'seed_11/test.csv', transform=eval_transform)
    sample = training[0]
    # sample: RGB image/transformed image, disease_id, background_id, image_id
    # For traditional features: use np.asarray(sample['image']).
    # For PyTorch DataLoader: supply a transform producing a fixed-size tensor.

Use deterministic image resizing and RGB conversion at loading time. Choose and
document the same ResNet input transform for both variants. Inspect the effect
of cropping on edge lesions and resizing on narrow leaves before freezing that
choice. Apply random augmentation to training samples only. Disease labels are
targets; background IDs remain evaluation metadata. Compute scalers, PCA,
feature selection, imputation or any dataset-derived normalisation only on each
training partition. Predefined ImageNet normalisation is not estimated here.
Do not remove backgrounds, denoise, threshold colours or crop lesions as a shared
default without an experiment supporting that choice. Such changes can remove
diagnostic information and alter the background comparison.

For frozen ResNet, model.train() recursively enables BatchNorm updates; set the
backbone back to eval() after that call. During layer4 fine-tuning, enforce the
chosen fixed BatchNorm running-statistics policy after model.train() as well.
Those training behaviours belong in the model code and are not implemented here.

VERIFICATION
    python -m unittest discover -s data_preprocessing -p 'test_prepare_dataset.py' -v
Tests cover seeded allocation, cross-background groups, infeasible coverage,
pending reviews, exact-copy conflicts, transitive deduplication, leakage detection
and RGB loader/hash behaviour. Model training and torch DataLoader integration
require the training environment and are outside this preprocessing verification.

Keep generated manifests, figures and thumbnails out of the submission archive
when required by the brief; provide the Python generator and reproduction steps.

CURRENT_SPLIT_VERSION.txt points to the delivered version. A future build prints
the new version path; update this pointer explicitly when the whole team adopts
that version. Never change a team's manifest version silently during experiments.
Recreate the inspection report with:
    python data_preprocessing/make_report.py Dhan-Shomadhan/preprocessing Dhan-Shomadhan/preprocessing/<version>
