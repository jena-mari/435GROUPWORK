# Role 4 supplied thumbnail SVM results

These files are preserved from the `CSCI435_Role4` handover. They document a
**preliminary thumbnail experiment**, not the full-resolution baseline or either
ResNet model. The source used fixed `C=1`, `gamma=scale` and balanced classes;
it did not run the baseline's full hyperparameter search.

- `test_predictions.csv`: five seeds combined, using the historical label schema.
- `metrics_by_run.csv`, `summary.csv`, `confusion_matrices.json` and
  `confusion_matrix_pooled.png`: supplied measured outputs.
- `Measured results.pdf` and `Evaluation notes.pdf`: supplied report drafts,
  preserved without edits. Their interpretation and section numbering need review
  before final submission. The repeated test splits are not independent datasets.
- `run_thumbnail_svm.py`: historical runner with hard-coded `/mnt/data` paths and
  an older import. Do not use it as the full-resolution baseline command.
- `role4_evaluation.py`: original supplied evaluator, retained for provenance.
  Use `python -m evaluation.role4_evaluation` for the integrated evaluator, which
  checks current canonical outputs and refuses incomplete comparisons.
- `README.txt`: original handover; its PDF filenames differ from those supplied.

`verification.json` records the manifest and metric checks made during integration.
The original supplied files retain their bytes. The evaluation package's README
explains how to recompute these results separately from the final comparison.
