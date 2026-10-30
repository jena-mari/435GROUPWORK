# Evaluation

`role4_evaluation.py` scores saved predictions. It does not train models or
choose checkpoints. Before writing any output it checks the seed, method,
manifest version, image IDs, labels and background IDs for every run.

Run it with the compact result folders:

```bash
python -m evaluation.role4_evaluation \
  --svm results/baseline \
  --frozen results/frozen_resnet18 \
  --finetuned results/finetuned_resnet18 \
  --out /tmp/rice-comparison
```

The output includes per-run metrics, five-run means and sample standard
deviations, per-class metrics, paired differences, prediction review rows and
confusion matrices. White, field and combined subsets are scored separately.
Combined metrics use every prediction in the test partition.

The compact comparison already generated for this project is under
`../results/comparison/`. The pooled matrices count test appearances across
overlapping holdouts, so they are diagnostic figures rather than one independent
test set.
