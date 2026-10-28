# Common CNN setup

These components implement the shared parts of the two ResNet experiments.
Role 5 owns the frozen-model experiment loop; Role 6 owns the fine-tuned loop.
The common checks do not replace review of those loops or the final five-run
evaluation.

## Environment and weights

Use Python 3.12 and `resnet18/requirements.txt` in `.venv-cnn`, separate from the
SVM environment. The tested versions are PyTorch 2.14.1, Torchvision 0.29.1,
NumPy 2.3.5, Pillow 12.3.0, scikit-learn 1.7.2 and SciPy 1.16.3.

`build_model('frozen', cache_dir=dataset_root/'model_cache/torchvision')` loads
the official `ResNet18_Weights.IMAGENET1K_V1` state and replaces its classifier
with `Linear(512, 5)`. Cached/downloaded files are checked against the official
SHA-256 prefix. An offline ImageNet state can be passed through `weights_path`;
it must match that checksum. Do not pass a project five-class checkpoint there.

The check selects CUDA or MPS if available, otherwise CPU. `--device cpu` is
available for diagnosing backend problems. Record the actual device; a seed does
not guarantee bitwise-identical training on different backends. The weights,
checkpoints and environments are local and are not committed to Git.

## Agreed image policy

`make_shared_data()` uses `RiceDataset` for EXIF orientation and RGB conversion,
then applies the same `ImagePolicy` to both CNN variants:

| Step | Training | Validation and test |
|---|---|---|
| Geometry | Bilinear resize preserving aspect ratio, centred padding to 224 x 224 | Same |
| Horizontal flip | Probability 0.5 | None |
| Rotation | Uniformly sampled from -10 to +10 degrees, with expanded canvas | None |
| After rotation | Resize/pad again to 224 x 224 so rotated edges are retained | Not needed |
| Tensor conversion | Float RGB tensor | Same |
| Normalisation | Mean `(0.485,0.456,0.406)`; SD `(0.229,0.224,0.225)` | Same |

Padding is RGB `(124,116,104)`, approximately the ImageNet mean. No colour jitter,
Gaussian blur, centre crop or dataset-fitted normalisation is applied. The
resize/padding policy is a project choice; it differs from the default centre
crop in the [Torchvision weight transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
The complete policy is part of checkpoint identity, so a mismatch is rejected.

`seed_everything()` seeds Python, NumPy and PyTorch. The shared data factory uses
seeded loader generators and worker initialisation. Start with `num_workers=0`.
Put executable code inside a main guard before adding worker processes on
macOS/Windows.

## Main APIs

| Function or class | Use |
|---|---|
| `make_shared_data(root, audit, seed)` | Verify manifest hashes, coverage and grouping; create the datasets, loaders and run context |
| `build_model('frozen', ...)` | Load pretrained weights, replace the head and set frozen-mode flags |
| `initialise_finetuned(frozen_checkpoint, context, device)` | Load the matching frozen selection and enable `layer4` plus the head |
| `prepare_training_mode(model, stage)` | Reapply backbone/BatchNorm modes after `model.train()` |
| `assert_model_contract(model, stage)` | Check stage structure, gradient flags and fixed BatchNorm statistics |
| `make_optimizer()` / `assert_optimizer_contract()` | Construct/check the stage's Adam groups and learning rates |
| `train_one_epoch()` / `evaluate()` | Common cross-entropy training and validation metrics |
| `BestCheckpoint` | Select highest validation macro-F1 and count consecutive non-improving epochs |
| `run_configuration()` | Produce JSON-ready settings, source hashes, versions and parameter groups |
| `load_checkpoint()` | Restore the selected state after checking context and stage |
| `export_predictions()` | Export predictions for the selected same-context checkpoint and exact image IDs |
| `score_predictions()` / `summarise_runs()` | Score background scenarios and summarise exactly five seeds |

The shared factory uses the existing manifests for seeds 11, 22, 33, 44 and 55.
It does not create new splits. Its `context` records the seed, manifest version,
all three manifest checksums, class/background mappings and image policy.

## Training and checkpoint contract

Frozen mode trains the **2,565 head parameters**. Fine-tuned mode trains
**8,396,293 parameters** in `layer4` and the head. Earlier stages stay fixed.
All BatchNorm running means, variances and batch counters stay fixed in both
stages. The BatchNorm affine parameters inside `layer4` are trainable during
fine-tuning.

The optimiser is Adam with betas `(0.9,0.999)`, epsilon `1e-8` and weight decay
zero. Frozen head LR is `1e-3`; fine-tuning uses `1e-5` for `layer4` and `1e-4`
for the head. Planned full runs use batch size 32, at most 30 frozen epochs and
20 additional fine-tuning epochs. Validation macro-F1 selects checkpoints; five
consecutive epochs without a strict improvement trigger stopping. Ties count
as no improvement. Fine-tuning starts a fresh selector and optimiser.

Cross-entropy consumes raw five-class logits. Do not add softmax to the forward
pass; the exporter applies it to generate probabilities. `train_one_epoch()`
rejects a validation/test loader and reapplies the model/BatchNorm policy.

Checkpoint loading rejects a different seed, manifest set, image policy, stage,
class mapping or full-experiment/check scope. Fine-tuned checkpoints also record
the parent frozen checkpoint's SHA-256 and epoch. Loading the frozen checkpoint
does not reuse its optimiser or early-stopping counter.

These utilities cover checkpoint selection and stage transfer. They do not
promise exact mid-epoch resumption of random-number streams.

## Prediction files

Load the selected checkpoint before export. The exporter checks its context and
stage, the loader seed, unique image IDs, exact coverage and label/background
agreement. Rows are returned in manifest order even with a shuffled loader.

```text
image_id,seed,method,disease_id,predicted_disease_id,background_id,prob_0,...,prob_4
```

The method is `frozen_resnet18` or `finetuned_resnet18`. A metadata sidecar records
the context and partition. Metrics use labels 0–4 and scenarios `field`, `white`
and `all`. Combined metrics are calculated from combined predictions.

`summarise_runs()` requires all five seeds and returns mean and sample SD
(`ddof=1`) separately for each scenario. Metrics are fractions: multiply both
means and SDs by 100 for percentage tables. Summarise each method separately
after checking that its runs use the same agreed version and policy.

## Small validation check

```bash
python -m resnet18.validate_implementation --dataset /path/to/Dhan-Shomadhan --seed 11
```

Defaults: one image per disease/background stratum in training and validation,
batch size 4, one epoch for each stage. It uses genuine pretrained weights and
starts fine-tuning from its selected frozen checkpoint. No test images are
evaluated. An explicit `--output` can select a fresh directory under the dataset.

The output is under `DATASET/experiments/resnet18_checks/seed_11/`: run settings,
per-stage histories, validation predictions/metrics, selected checkpoints and
`implementation_check.json`. Small-check checkpoints cannot be loaded using a
full-experiment context or used to export test predictions. Their scores are
diagnostic, not final report results.

Full experiments remain under `experiments/frozen_resnet18/seed_<seed>/` and
`experiments/finetuned_resnet18/seed_<seed>/`. See the
[role checklist](ROLE_VALIDATION.md) before starting them.
