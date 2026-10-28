# ResNet-18 methods

This folder is reserved for the frozen and partly fine-tuned ResNet-18
implementations. **Training code, CNN-generated figures and CNN results have not
been added yet.** The settings below are the project plan, not completed runs.

Both variants must read the same `RiceDataset` loader, disease mapping and
current manifest version used by the HSV–LBP–SVM baseline. The input transform,
normalisation and training augmentation policy still need to be recorded in the
training implementation. SVM resizing and blurring are specific to that method.

| Variant | Starting checkpoint | Trainable parameters | Planned settings | Local output directory |
|---|---|---|---|---|
| Frozen | ImageNet-pretrained ResNet-18 | Replacement five-class linear head | Adam, head LR `1e-3`, batch size 32, up to 30 epochs | `Dhan-Shomadhan/experiments/frozen_resnet18/` |
| Fine-tuned | Best frozen checkpoint from the same seed | `layer4` and the five-class head | Adam, `layer4` LR `1e-5`, head LR `1e-4`, batch size 32, up to 20 additional epochs | `Dhan-Shomadhan/experiments/finetuned_resnet18/` |

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
