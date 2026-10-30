# Dataset directory

The original Dhan-Shomadhan images are local inputs and are not committed to
Git. Download version 1 from the Mendeley record linked in the project README,
then preserve these source folders exactly:

```text
Dhan-Shomadhan/
├── Field Background  /
├── White Background /
└── Dhan-Shomadhan_picture_Information.csv
```

The spaces at the ends of the folder names are part of the original data. The
audit and manifests are already committed under `preprocessing/`.

| Local path | Use |
|---|---|
| `preprocessing/` | Audit records, thumbnails, review files and shared manifests |
| `svm_cache/` | Local HSV/LBP features; not committed |
| `experiments/` | Local checkpoints and full run files; compact copies are in the root `results/` folder |
| `model_cache/torchvision/` | Official ImageNet weights; not committed |

The current split pointer is in `preprocessing/CURRENT_SPLIT_VERSION.txt`.
All methods must use that version. The saved partitions are repeated holdouts
with group separation and target proportions of 70/15/15.
