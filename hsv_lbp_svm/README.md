# HSV–LBP–SVM baseline

The baseline uses the shared manifests and the original group feature design:
256×256 resize, 3×3 Gaussian blur, a 16×4×4 HSV histogram and uniform LBP
histograms at `(8, 1)` and `(16, 2)`. The resulting 284 features are scaled on
training data only. An RBF SVM searches the recorded C/gamma grid and selects
the highest validation macro-F1.

Run a validation-only check:

```bash
python -m hsv_lbp_svm.svm_baseline \
  --dataset /path/to/Dhan-Shomadhan --seeds 11 --validation-only
```

The full completed results are in `../results/baseline/`. The selected model
pickles and feature cache stay local. See `../GROUP_HANDOVER.md` for the files
needed by the results section.
