# Role 5 and Role 6 integration checks

The common code is checked. The separate role implementations have not yet been
supplied for review. Use this checklist to validate those loops before the final
five-run test evaluation.

## Role 5: frozen model

1. Call `seed_everything(seed)` before initialising the head.
2. Create data with `make_shared_data(dataset_root, audit_directory, seed)`.
3. Use `build_model('frozen', cache_dir=dataset_root/'model_cache/torchvision')`.
4. Confirm `assert_model_contract(model, 'frozen')` reports 2,565 trainable
   parameters and `assert_optimizer_contract()` passes.
5. Train for up to 30 epochs, selecting by validation macro-F1 with patience 5.
6. Load the selected checkpoint before exporting predictions.

With `dataset_root`, `audit_directory`, `output` and `seed` defined as Paths and
an integer, the loop can use:

```python
from resnet18.common import (
    STAGE_DEFAULTS, BestCheckpoint, build_model, choose_device, evaluate,
    load_checkpoint, make_optimizer, make_shared_data, run_configuration,
    seed_everything, train_one_epoch,
)
from data_preprocessing.prepare_dataset import write_csv, write_json

seed_everything(seed)
data = make_shared_data(dataset_root, audit_directory, seed, batch_size=32)
device = choose_device()
stage = 'frozen'
model = build_model(stage, cache_dir=dataset_root/'model_cache/torchvision').to(device)
optimizer = make_optimizer(model, stage)
output.mkdir(parents=True, exist_ok=True)
config = run_configuration(model, optimizer, data, stage, device)
write_json(output/'run_config.json', config)
selector = BestCheckpoint(output/'best.pt', data.context, stage, patience=5)
history = []

for epoch in range(1, STAGE_DEFAULTS[stage]['max_epochs'] + 1):
    train = train_one_epoch(model, data.loaders['train'], optimizer, device, stage)
    val = evaluate(model, data.loaders['validation'], device)
    history.append({'epoch': epoch, 'training_loss': train['loss'],
                    'validation_loss': val['loss'], 'validation_macro_f1': val['macro_f1']})
    if selector.update(val['macro_f1'], epoch, model, optimizer, config):
        break

write_csv(output/'validation_history.csv', history)
load_checkpoint(output/'best.pt', model, data.context, required_stage=stage)
```

Use a fresh output directory for a new run. Record any actual overrides to the
planned defaults in `run_config.json` and the report.

## Role 6: fine-tuned model

Use the same data factory and replace the initialisation above with:

```python
from resnet18.common import initialise_finetuned

stage = 'finetuned'
model, frozen_metadata = initialise_finetuned(frozen_checkpoint, data.context, device)
optimizer = make_optimizer(model, stage)
```

Create a new `BestCheckpoint` for this stage and reset its patience counter.
Use at most 20 additional epochs. Do not carry over the frozen optimiser or best
validation score. The source must be the selected frozen checkpoint for this
seed, manifest version and input policy. The helper records its SHA-256 and epoch.

Confirm that only `layer4` and the head are trainable, with learning rates
`1e-5` and `1e-4`, respectively. After a step, at least one parameter in each
component should change; earlier parameters and all BatchNorm buffers should
stay identical. The small diagnostic runner performs this check.

## Shared checks before a full run

- Verify the original image files using `data_preprocessing/verify_dataset.py`.
- Inspect a loader batch: float images `[B,3,224,224]`, integer disease labels
  0–4 and matching image IDs. Background remains metadata.
- Use `prepare_training_mode()` after any custom `model.train()` call. Freezing
  gradients alone does not freeze BatchNorm statistics.
- Use the same transforms for both variants, with random augmentation only in
  the training loader. Keep `drop_last=False` for evaluation.
- Retain validation histories and choose checkpoints without test scores.
- Check the actual batch size, epoch cap, Adam settings, seed, package versions
  and device in each run configuration.

## Export and score

After the procedures are fixed and validation selection is complete:

```python
from resnet18.common import export_predictions, score_predictions

predictions = export_predictions(
    model, data.loaders['test'], output/'test_predictions.csv',
    data.context, stage, device,
)
write_json(output/'metrics.json', score_predictions(predictions))
```

Export requires the selected checkpoint to have been loaded. It rejects an
incorrect seed or stage and catches missing/repeated image IDs and mismatched
labels. See [COMMON_COMPONENTS.md](COMMON_COMPONENTS.md) for the output columns
and five-seed summary helper. Keep every method's runs in its own output folder.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing source files | Extraction location and trailing spaces; verify the dataset rather than editing manifests |
| PIL batch-collation error | Apply the common transform through `RiceDataset` so it returns fixed-size tensors |
| Wrong loss input | Use five raw logits and integer labels on the same device; do not apply softmax before cross-entropy |
| Frozen representation changes | Inspect gradient flags and BatchNorm buffers; reapply the common training mode |
| No `layer4` update | Check fine-tuned initialisation, gradient flags and optimiser groups |
| Checkpoint mismatch | Check seed, manifests, stage, image policy and diagnostic/full-run scope; do not bypass the check |
| Weight-cache checksum failure | Replace the damaged official weight file; distinguish ImageNet weights from project checkpoints |
| Missing predictions | Keep `drop_last=False`, match by image ID and use the exporter |
| MPS/backend error | Repeat the small check with `--device cpu` and compare the failure before changing model settings |
| Memory pressure | Start with the small batch-4 check; record any batch-size change for a full experiment |
| Low small-check validation score | Ten validation images and one epoch check execution, not final performance |

Place executable loops under a main guard before enabling loader worker
processes on macOS/Windows. Fixed seeds reproduce partition membership and help
repeat training; they do not promise identical numerical results across backends.
