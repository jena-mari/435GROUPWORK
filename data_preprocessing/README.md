# Shared data preparation

The preprocessing code audits the downloaded images, checks labels and hashes,
records possible duplicates, and builds the five shared group-preserving
train/validation/test splits.

Run the standard checks from the repository root:

```bash
python data_preprocessing/verify_dataset.py \
  --root /path/to/Dhan-Shomadhan \
  --audit Dhan-Shomadhan/preprocessing
python -m unittest discover -s data_preprocessing -p 'test_*.py' -v
```

The current version is the path in
`Dhan-Shomadhan/preprocessing/CURRENT_SPLIT_VERSION.txt`. Do not generate a new
split for an individual method. Background and image IDs are metadata used for
analysis and prediction checks; they are not classifier inputs.

The audit found no unreadable files or byte-identical duplicates. Pending
perceptual-hash candidates were grouped conservatively to reduce leakage risk.
This can over-group unrelated leaves, so the limitation should be stated in the
report.
