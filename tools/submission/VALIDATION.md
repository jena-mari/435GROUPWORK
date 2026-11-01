# Standalone submission validation

Validated on 1 November 2026 with the existing Python 3.12 SVM and CNN environments.

- Six packaging/routing tests pass: embedded-source hashes, original audit
  checksums, non-overwrite behaviour, dataset-contained outputs, unique run
  directories, environment selection and five-seed dispatch.
- The portable preliminary thumbnail runner passes a five-seed synthetic-image
  test covering actual HSV/LBP extraction, scaling, SVM fitting and historical
  prediction export. It uses no pandas or `/mnt/data` paths.
- With only `group_name.py` in an isolated directory, setup is repeatable and
  the report comparison runs without source images or a repository checkout.
  Both repeated evaluations reproduce all nine recorded summary rows.
- The supplied preliminary thumbnail predictions produce their separate
  three-scenario comparison.
- All 1,106 real source images, the metadata and the original 15 manifests pass
  checksum verification through the standalone file.
- The bundled CNN implementation check passes loader/transform, verified
  weights, freezing, BatchNorm, parameter updates, prediction and checkpoint checks.
- A one-epoch, eight-image validation-only smoke run completes.
- Full SVM training across all five shared seeds completes; each exported
  per-image prediction file exactly matches the repository's recorded result.
- A small real-image run of the bundled frozen and fine-tuned loops completes
  checkpoint selection, both learning-rate candidates and held-out prediction
  export. It is a diagnostic, not a replacement for the report's results.
- The installed final entry point also regenerates the nine report rows and
  confusion-matrix figures under `Dhan-Shomadhan/experiments/`.

Full five-seed CNN retraining, the ten-epoch resize comparison and full thumbnail
retraining were not rerun as part of this validation. The main CNN source modules
are byte-identical to the current repository implementations and their original
training/selection settings are retained.
