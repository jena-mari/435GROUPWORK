# ResNet-18 experiments and shared components

This folder provides the common components for **Role 5 (frozen model)** and
**Role 6 (fine-tuned model)**. Pretrained loading, preprocessing, augmentation,
model/optimiser checks, epoch helpers, checkpoints and prediction export are
implemented. Role 5's frozen experiment is integrated in `train_frozen.py`.
The fine-tuned experiment driver and final five-run test results are still pending.

| File | Purpose |
|---|---|
| `train_frozen.py` | Frozen experiment using the current shared components |
| `ROLE5_FROZEN_EXPERIMENTS.md` | Training, validation-only and test-export commands |
| `common.py` | Shared model, data, training, checkpoint and export helpers |
| `validate_implementation.py` | Small pretrained training/validation check for both modes |
| `tests/test_common.py` | Offline unit and integration checks with synthetic images |
| `requirements.txt` | Tested CNN environment |
| `COMMON_COMPONENTS.md` | Agreed image policy, APIs and output contracts |
| `ROLE_VALIDATION.md` | Integration examples and troubleshooting for Roles 5 and 6 |

From the repository root, use a separate Python 3.12 environment:

```bash
python3.12 -m venv .venv-cnn
source .venv-cnn/bin/activate
python -m pip install -r resnet18/requirements.txt
python -m unittest discover -s resnet18/tests -p 'test_*.py' -v
python -m resnet18.validate_implementation --dataset /path/to/Dhan-Shomadhan
```

A `.venv-cnn` is already set up in the original checkout. On Windows use
`py -3.12` and `.venv-cnn\Scripts\activate`. Full instructions are in
[COMMON_COMPONENTS.md](COMMON_COMPONENTS.md). The diagnostic uses 10 training
and 10 validation images by default and never evaluates test images.

Both variants must read the same `RiceDataset` loader, disease mapping and
current manifest version used by the HSV–LBP–SVM baseline. The agreed policy is
224 x 224 aspect-preserving resize/padding, ImageNet normalisation, and training-only
horizontal flips and rotations up to 10 degrees. Both variants use this same
policy, recorded in each checkpoint. SVM resizing and blurring are specific to
that method.

| Variant | Starting checkpoint | Trainable parameters | Planned settings | Local output directory |
|---|---|---|---|---|
| Frozen | ImageNet-pretrained ResNet-18 | Replacement five-class linear head | Adam, head LR `1e-3`, batch size 32, up to 30 epochs | `Dhan-Shomadhan/experiments/frozen_resnet18/` |
| Fine-tuned | Best frozen checkpoint from the same seed | `layer4` and the five-class head | Adam, `layer4` LR `1e-5`, head LR `1e-4`, batch size 32, up to 20 additional epochs | `Dhan-Shomadhan/experiments/finetuned_resnet18/` |

Follow [the frozen experiment guide](ROLE5_FROZEN_EXPERIMENTS.md) to run Role 5's
loop and pass its selected `best.pt` checkpoints to Role 6. Use the
[evaluation package](../evaluation/README.md) once saved test predictions are ready.

Use validation macro-F1 for checkpoint selection, with five consecutive epochs
without improvement as the stopping rule. Keep the agreed BatchNorm running
statistics fixed, including after calls to `model.train()`. Evaluate test images
only after the training and selection procedure is fixed.

Each variant needs runs for seeds 11, 22, 33, 44 and 55. For every seed, return
the configuration, validation history, selected checkpoint location and test
predictions in the format described in the [group handover](../GROUP_HANDOVER.md).
Keep model checkpoints and temporary arrays local.

Future learning curves, confusion matrices or feature visualisations should go
under that variant's `Dhan-Shomadhan/experiments/<method>/figures/` directory and
identify their seed, background subset and source run. They should not be mixed
with preprocessing sample images or SVM figures.

The [shared preprocessing guide](../data_preprocessing/README.md) explains the
loader and splits. Preserve the current partitions when adding the CNN code.
