# Building the standalone submission

The submitted `CSCI435Group2/group_name.py` is generated from the repository's
actual Python modules, shared audit records and compact recorded predictions.
No source images, checkpoints, environments, raster previews or PDFs enter the
payload. The entry-point template remains readable here; the payload preserves
the original bytes and SHA-256 checksums of the source modules and manifests.

From the repository root:

```bash
python3.12 tools/submission/build_submission.py --repo . --output CSCI435Group2/group_name.py
python3.12 -m unittest discover -s tools/submission -p test_submission.py -v
```

The builder exports all original runtime modules. The portable thumbnail runner
replaces only the legacy `/mnt/data`-specific thumbnail entry point and avoids
its pandas dependency; the historical source is still retained in the runtime.
The three main training methods execute the repository implementations without
altering their feature extraction, architecture, split allocation or selection
rules. The complete original five-seed evaluation is also embedded so the marker
can reconstruct the report without needing the GitHub checkout.

The submission user manual is `CSCI435Group2/USER_MANUAL.md`. The Moodle ZIP
should contain only `group_name.py` and the final report `group_name.pdf`.
