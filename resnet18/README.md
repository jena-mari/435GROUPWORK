# ResNet-18

This folder contains the shared CNN code and the two experiment drivers.

| File | Purpose |
|---|---|
| `common.py` | Data factory, transforms, model contracts, checkpoints and export |
| `train_frozen.py` | Frozen ImageNet ResNet-18 experiment |
| `train_finetuned.py` | Same-seed `layer4` fine-tuning experiment |
| `check_cnn.py` | Weight check, transform preview and small diagnostics |
| `validate_implementation.py` | Two-stage implementation check |
| `SHARED_COMPONENTS_INTEGRATION.md` | API mapping and preprocessing decisions |
| `ROLE5_FROZEN_EXPERIMENTS.md` | Frozen-model commands and output files |

Both CNNs use 224×224 aspect-preserving padding, ImageNet normalisation,
training-only horizontal flips and rotations up to ±10°. BatchNorm running
statistics are held fixed. The frozen model trains only its 2,565-parameter
head. Fine-tuning starts from the matching frozen checkpoint and trains
`layer4` plus the head.

The completed compact outputs are in `../results/frozen_resnet18/` and
`../results/finetuned_resnet18/`. Checkpoints remain local because each is large.
See `../GROUP_HANDOVER.md` before adding a new run.
