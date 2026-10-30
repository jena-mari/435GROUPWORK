"""Shared ResNet-18 components for the frozen and fine-tuned CNN methods (Role 3).

Both CNN variants import this module so that they use the same:
    * shared splits, loaded through data_preprocessing's RiceDataset   load_shared_splits()
    * CNN input transforms passed through RiceDataset(transform=...)    TrainTransform, EvalTransform
    * data loaders with reproducible augmentation                       make_loader()
    * ImageNet ResNet-18 weights, verified on load                      build_resnet18(), download_weights()
    * layer freezing and fixed BatchNorm statistics                     set_trainable(), ResNet18.train()
    * checkpoint saving, loading and selection                          save_checkpoint(), load_checkpoint(), BestTracker
    * prediction, export in the group format and scoring                predict(), write_predictions(), get_scores()
    * run records                                                       new_run_dir(), base_run_config(), summary_rows()

Run from the repository root. Permitted packages only (PyTorch, NumPy, Pillow, scikit-learn); ResNet-18 is
defined here in plain PyTorch, so torchvision is not needed. The SVM's resize and Gaussian blur are not used.

Minimal use (Role 5 shown; Role 6 starts from the frozen checkpoint and trains {"layer4": 1e-5, "fc": 1e-4}):

    from resnet18 import common as cc
    cc.set_seed(seed)
    data, version = cc.load_shared_splits(cc.DATASET, cc.AUDIT, seed)        # RiceDataset objects
    train_dl = cc.make_loader(data["train"], train=True, seed=seed)
    val_dl = cc.make_loader(data["validation"], train=False, seed=seed)
    model = cc.build_resnet18(cc.DATASET).to(device)
    optimiser = torch.optim.Adam(cc.set_trainable(model, {"fc": 1e-3}))
    for epoch ...:
        model.train()                                  # BatchNorm statistics stay fixed automatically
        ...
        val = cc.predict(model, val_dl, device)
        if tracker.update(val["macro_f1"], val["loss"], epoch): cc.save_checkpoint(...)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import PIL
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data_preprocessing.prepare_dataset import (
    BACKGROUNDS, CLASSES, PARTITIONS, SEEDS, RiceDataset, rgb_image, sha_file, validate_split, write_csv, write_json,
)

# ============================================================================================
# Shared locations and agreed settings
# ============================================================================================
REPO = Path(__file__).resolve().parents[1]
DATASET = Path("Dhan-Shomadhan")
AUDIT = REPO / "Dhan-Shomadhan" / "preprocessing"
NUM_CLASSES = len(CLASSES)
METHODS = ("frozen_resnet18", "finetuned_resnet18")

WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_FILE = Path("pretrained") / "resnet18-f37072fd.pth"       # under the dataset directory
WEIGHTS_SHA256_PREFIX = "f37072fd"                                  # PyTorch's published hash prefix

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FILL = tuple(int(round(255 * m)) for m in IMAGENET_MEAN)           # padding colour; becomes ~0 after normalising

# Input-size policy. Source images are portrait (1952 x 4160 for 1,089 of 1,106 images, ratio ~2.13).
#   portrait: turn landscape images upright, resize to 448 x 224 (H x W). Keeps the leaf's shape and gives
#             lesions twice the pixels of a square input; ResNet's global pooling accepts non-square inputs.
#   squash:   resize to 224 x 224 (stretches leaves about 2x sideways).
#   pad:      keep aspect ratio, pad to 224 x 224 with the ImageNet mean colour (photo ~105 px wide).
RESIZE_MODES = {"portrait": (448, 224), "squash": (224, 224), "pad": (224, 224)}
RESIZE_MODE = "portrait"            # proposed default; confirm with check_cnn compare-resize on validation data

AUGMENTATION = dict(crop_area=(0.6, 1.0), crop_aspect_jitter=(3 / 4, 4 / 3), hflip=0.5, vflip=0.5,
                    rotation_degrees=15.0, brightness=0.2, contrast=0.2, hue_saturation="unchanged")
BATCH_SIZE = 32
PATIENCE = 5
SELECTION_RULE = ("highest validation macro-F1; exact ties keep the lower validation loss; "
                  f"stop after {PATIENCE} consecutive epochs without improvement; test images not used")


# ============================================================================================
# Reproducibility and environment
# ============================================================================================
def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(force_cpu=False):
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    if not force_cpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment(device):
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                pillow=PIL.__version__, sklearn=sklearn.__version__, device=str(device),
                cuda=torch.version.cuda if torch.cuda.is_available() else None,
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark))


# ============================================================================================
# Shared splits through RiceDataset (same checks as the SVM baseline's loader)
# ============================================================================================
def source_path(root, relative):
    """Full path to a source image. resolve() matters on Windows: the source folders end in spaces
    ("Field Background  ", "White Background "), and Windows drops trailing spaces from a plain path, so the
    file cannot be opened. The resolved path keeps them (the same form RiceDataset uses to check the files)."""
    return (Path(root) / relative).resolve()


class SharedRiceDataset(RiceDataset):
    """The group's RiceDataset with all its manifest checks, unchanged, except that each image is opened
    through its resolved path (see source_path) so loading also works on Windows. Returns the same fields."""

    def __getitem__(self, index):
        row = self.rows[index]
        image = rgb_image(source_path(self.root, row["path"]))
        if self.transform is not None:
            image = self.transform(image)
        return dict(image=image, disease_id=int(row["disease_id"]),
                    background_id=int(row["background_id"]), image_id=row["image_id"])


def current_version(audit=AUDIT):
    audit = Path(audit).resolve()
    version = (audit / (audit / "CURRENT_SPLIT_VERSION.txt").read_text().strip()).resolve()
    if not version.is_relative_to(audit):
        raise ValueError("Split version must be inside the shared audit directory")
    return version


def load_shared_splits(root=DATASET, audit=AUDIT, seed=11, train_transform=None, eval_transform=None,
                       resize_mode=RESIZE_MODE):
    """RiceDataset objects for one seed's train/validation/test manifests, with CNN transforms attached.
    Checks manifest checksums, seed/partition fields, group separation and full coverage."""
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    train_transform = train_transform or TrainTransform(seed, resize_mode)
    eval_transform = eval_transform or EvalTransform(resize_mode)
    version = current_version(audit)
    provenance = json.loads((version / "provenance.json").read_text())
    datasets, all_rows, assignment = {}, [], {}
    for partition in PARTITIONS:
        relative = f"seed_{seed}/{partition}.csv"
        manifest = version / relative
        if sha_file(manifest) != provenance["manifest_sha256"][relative]:
            raise ValueError(f"Shared manifest checksum changed: {relative}")
        transform = train_transform if partition == "train" else eval_transform
        dataset = SharedRiceDataset(root, manifest, transform=transform)
        for row in dataset.rows:
            if int(row["seed"]) != seed or row["partition"] != partition:
                raise ValueError("Manifest seed/partition mismatch")
            gid = row["group_id"]
            if gid in assignment and assignment[gid] != partition:
                raise ValueError("A related-image group crosses partitions")
            assignment[gid] = partition
        datasets[partition] = dataset
        all_rows.extend(dataset.rows)
    validate_split(all_rows, assignment)
    if len(all_rows) != provenance["eligible_images"]:
        raise ValueError("Shared partitions do not cover the eligible inventory")
    return datasets, version


def manifest_version_name(version, audit=AUDIT):
    return Path(version).resolve().relative_to(Path(audit).resolve()).as_posix()


# ============================================================================================
# CNN transforms (passed to RiceDataset's transform= argument; input is an RGB PIL image)
# ============================================================================================
def _orient(img, mode):
    """Portrait mode turns the few landscape images upright; leaves have no fixed orientation."""
    if mode == "portrait" and img.width > img.height:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img


def _fit(img, mode):
    """Resize a PIL image to the model input size according to the resize mode."""
    h, w = RESIZE_MODES[mode]
    if mode != "pad":
        return img.resize((w, h), Image.Resampling.BILINEAR, reducing_gap=3.0)
    scale = min(w / img.width, h / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    canvas = Image.new("RGB", (w, h), FILL)
    canvas.paste(img.resize((nw, nh), Image.Resampling.BILINEAR, reducing_gap=3.0), ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _random_crop(img, rng):
    """Random region of 60-100% of the image area, aspect ratio jittered by 3:4-4:3 relative to the image."""
    W, H = img.size
    lo, hi = AUGMENTATION["crop_aspect_jitter"]
    for _ in range(10):
        area = rng.uniform(*AUGMENTATION["crop_area"])
        r = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        cw, ch = round(W * math.sqrt(area * r)), round(H * math.sqrt(area / r))
        if 0 < cw <= W and 0 < ch <= H:
            x0, y0 = int(rng.integers(0, W - cw + 1)), int(rng.integers(0, H - ch + 1))
            return img.crop((x0, y0, x0 + cw, y0 + ch))
    return img


def normalise(img):
    """PIL RGB -> float32 HxWx3 array with fixed ImageNet normalisation (no statistics from our data)."""
    return (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def to_tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class EvalTransform:
    """Deterministic: orient, resize the whole image (no cropping, so leaf tips are kept), normalise."""

    def __init__(self, resize_mode=RESIZE_MODE):
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {list(RESIZE_MODES)}")
        self.mode = resize_mode

    def pil(self, img):
        return _fit(_orient(img, self.mode), self.mode)

    def __call__(self, img):
        return to_tensor(normalise(self.pil(img)))

    def settings(self):
        return dict(type="eval", resize_mode=self.mode, input_hw=RESIZE_MODES[self.mode], resample="bilinear",
                    crop="none (whole image)", normalisation=dict(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()))


class TrainTransform(EvalTransform):
    """Training only: random crop, resize, flips, rotation, brightness/contrast, then normalise.
    Uses its own seeded generator; make_loader() reseeds it per worker so runs are reproducible."""

    def __init__(self, seed, resize_mode=RESIZE_MODE):
        super().__init__(resize_mode)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def pil(self, img):
        rng, a = self.rng, AUGMENTATION
        img = _fit(_random_crop(_orient(img, self.mode), rng), self.mode)
        if rng.random() < a["hflip"]:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rng.random() < a["vflip"]:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        img = img.rotate(rng.uniform(-a["rotation_degrees"], a["rotation_degrees"]),
                         resample=Image.Resampling.BILINEAR, fillcolor=FILL)
        img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-a["brightness"], a["brightness"]))
        img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-a["contrast"], a["contrast"]))
        return img

    def settings(self):
        s = super().settings()
        s.update(type="train", seed=self.seed, augmentation=AUGMENTATION, order=[
            "orient", "random crop", "resize", "flips", "rotation (ImageNet-mean fill)", "brightness", "contrast",
            "normalise"])
        return s


def transform_settings(datasets):
    return {part: ds.transform.settings() for part, ds in datasets.items()}


# ============================================================================================
# Loaders
# ============================================================================================
def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    transform = getattr(info.dataset, "transform", None)
    if hasattr(transform, "reseed"):
        transform.reseed(torch.initial_seed() % 2 ** 32)   # derived from the loader generator: reproducible


def make_loader(dataset, train, seed, batch_size=BATCH_SIZE, workers=0):
    """train=True: shuffled (seeded) training batches; otherwise fixed order.
    Batches are dicts: image (B x 3 x H x W), disease_id, background_id (tensors), image_id (list of str).
    workers > 0 speeds up decoding of the full-size JPEGs; on Windows call from under `if __name__ == "__main__":`."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed),
                      worker_init_fn=_worker_init if workers else None, persistent_workers=bool(workers))


# ============================================================================================
# ResNet-18 (layer names match PyTorch's ImageNet weights file)
# ============================================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + identity)


class ResNet18(nn.Module):
    STAGES = ("stem", "layer1", "layer2", "layer3", "layer4", "fc")

    def __init__(self, num_classes=NUM_CLASSES, freeze_bn_stats=True):
        super().__init__()
        self.freeze_bn_stats = freeze_bn_stats
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.fc = nn.Linear(512, num_classes)

    def train(self, mode=True):
        """model.train() normally puts BatchNorm into training mode, which updates its running statistics.
        With freeze_bn_stats (the agreed policy) every BatchNorm layer is put straight back into eval mode,
        so the ImageNet running mean/variance never change. BN scale/shift of unfrozen stages still learn."""
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def stage(self, name):
        return nn.ModuleList([self.conv1, self.bn1]) if name == "stem" else getattr(self, name)

    def features(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)

    def forward(self, x):
        return self.fc(self.features(x))


def weights_path(root=DATASET):
    return Path(root) / WEIGHTS_FILE


def download_weights(root=DATASET):
    dst = weights_path(root)
    if dst.exists():
        print(f"Weights already present: {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {WEIGHTS_URL}\n  -> {dst}")
        torch.hub.download_url_to_file(WEIGHTS_URL, str(dst), progress=True)
    verify_weights(dst)
    return dst


def verify_weights(path):
    digest = sha_file(path)
    if not digest.startswith(WEIGHTS_SHA256_PREFIX):
        raise ValueError(f"{path} SHA-256 {digest[:12]}... does not match PyTorch's published prefix "
                         f"{WEIGHTS_SHA256_PREFIX}; download it again")
    return digest


def build_resnet18(root=DATASET, pretrained=True):
    """ResNet-18 with verified ImageNet backbone weights and a new randomly initialised 512 -> 5 head."""
    model = ResNet18(NUM_CLASSES)
    if not pretrained:
        return model
    path = weights_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python -m resnet18.check_cnn download-weights`, or "
                                f"download {WEIGHTS_URL} and place it there without renaming.")
    verify_weights(path)
    state = {k: v for k, v in torch.load(path, map_location="cpu", weights_only=True).items()
             if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise RuntimeError(f"Weights do not match ResNet-18: missing={missing}, unexpected={unexpected}")
    return model


def weights_record(root=DATASET):
    path = weights_path(root)
    return dict(source=WEIGHTS_URL, file=path.as_posix(), sha256=sha_file(path) if path.exists() else None)


def set_trainable(model, learning_rates):
    """Freeze every parameter, then unfreeze the named stages. Returns Adam parameter groups.
    learning_rates: {stage: lr}, stages from ResNet18.STAGES. Frozen: {"fc": 1e-3}; fine-tuned: {"layer4": 1e-5, "fc": 1e-4}."""
    for p in model.parameters():
        p.requires_grad = False
    groups = []
    for name, lr in learning_rates.items():
        params = list(model.stage(name).parameters())
        for p in params:
            p.requires_grad = True
        groups.append(dict(params=params, lr=lr, name=name))
    return groups


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable)


# ============================================================================================
# Checkpoints and selection
# ============================================================================================
def save_checkpoint(path, model, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), class_mapping=CLASSES, meta=meta), path)
    return path


def load_checkpoint(path, model=None, device="cpu"):
    """Returns (model, meta). Refuses checkpoints saved with a different disease-ID mapping."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("class_mapping") != CLASSES:
        raise ValueError(f"{path} uses class mapping {ckpt.get('class_mapping')}, expected {CLASSES}")
    model = model or ResNet18(NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt.get("meta", {})


class BestTracker:
    """Agreed rule: highest validation macro-F1; exact ties keep the lower validation loss;
    stop after `patience` consecutive epochs without improvement."""

    def __init__(self, patience=PATIENCE):
        self.patience, self.best_f1, self.best_loss, self.best_epoch, self.stale = patience, -1.0, math.inf, 0, 0

    def update(self, val_f1, val_loss, epoch):
        """True when this epoch is the new best (save the checkpoint then)."""
        if val_f1 > self.best_f1 or (val_f1 == self.best_f1 and val_loss < self.best_loss):
            self.best_f1, self.best_loss, self.best_epoch, self.stale = val_f1, val_loss, epoch, 0
            return True
        self.stale += 1
        return False

    @property
    def should_stop(self):
        return self.stale >= self.patience


# ============================================================================================
# Prediction, export and scoring (group handover format)
# ============================================================================================
def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))


@torch.no_grad()
def predict(model, loader, device="cpu"):
    """Per-image predictions keyed by image_id (safe with shuffled loaders), plus loss, accuracy, macro-F1."""
    model.eval()
    rows, loss_sum = [], 0.0
    for batch in loader:
        y = batch["disease_id"].to(device)
        logits = model(batch["image"].to(device))
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        probs = F.softmax(logits, 1).cpu().numpy()
        for iid, t, b, p in zip(batch["image_id"], batch["disease_id"].tolist(), batch["background_id"].tolist(), probs):
            rows.append(dict(image_id=iid, disease_id=int(t), predicted_disease_id=int(p.argmax()),
                             background_id=int(b), probs=p))
    y_true = [r["disease_id"] for r in rows]
    y_pred = [r["predicted_disease_id"] for r in rows]
    return dict(rows=rows, loss=loss_sum / len(rows), accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=macro_f1(y_true, y_pred))


def check_predictions(rows, dataset):
    """Every manifest image predicted exactly once, nothing extra, true labels/backgrounds match the manifest."""
    manifest = {r["image_id"]: r for r in dataset.rows}
    ids = [r["image_id"] for r in rows]
    repeated = {i for i in ids if ids.count(i) > 1}
    missing, extra = set(manifest) - set(ids), set(ids) - set(manifest)
    wrong = [r["image_id"] for r in rows if r["image_id"] in manifest and (
        r["disease_id"] != int(manifest[r["image_id"]]["disease_id"]) or
        r["background_id"] != int(manifest[r["image_id"]]["background_id"]))]
    if repeated or missing or extra or wrong:
        raise ValueError(f"Prediction check failed: {len(missing)} missing, {len(repeated)} repeated, "
                         f"{len(extra)} outside the manifest, {len(wrong)} label/background mismatches")
    return True


def write_predictions(path, rows, seed, method):
    """Group format: image_id,seed,method,disease_id,predicted_disease_id,background_id,prob_0..prob_4.
    Rows are sorted by image_id; results are matched on image_id, never on row order."""
    out = [dict(image_id=r["image_id"], seed=seed, method=method, disease_id=r["disease_id"],
                predicted_disease_id=r["predicted_disease_id"], background_id=r["background_id"],
                **{f"prob_{k}": f"{r['probs'][k]:.6f}" for k in range(NUM_CLASSES)})
           for r in sorted(rows, key=lambda r: r["image_id"])]
    write_csv(path, out)
    return path


def get_scores(rows):
    """metrics.json content, same layout as the SVM baseline: field, white and all test images, each computed
    from its own predictions (combined macro-F1 is not an average of the two)."""
    y = np.array([r["disease_id"] for r in rows])
    p = np.array([r["predicted_disease_id"] for r in rows])
    bg = np.array([r["background_id"] for r in rows])
    scores = {}
    for name in ("field", "white", "all"):
        mask = np.ones(len(y), bool) if name == "all" else bg == BACKGROUNDS[name]
        if not mask.any():
            raise ValueError(f"No test images for scenario {name}")
        scores[name] = dict(
            images=int(mask.sum()), accuracy=float(accuracy_score(y[mask], p[mask])), macro_f1=macro_f1(y[mask], p[mask]),
            per_class_f1=f1_score(y[mask], p[mask], average=None, labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
            confusion_matrix=confusion_matrix(y[mask], p[mask], labels=list(range(NUM_CLASSES))).tolist())
    return scores


def summary_rows(results):
    """Mean and sample standard deviation (ddof=1) over runs, per scenario (same layout as the SVM summary)."""
    rows = []
    for scenario in ("field", "white", "all"):
        acc = [r[scenario]["accuracy"] for r in results]
        f1 = [r[scenario]["macro_f1"] for r in results]
        rows.append(dict(scenario=scenario, runs=len(results), accuracy_mean=float(np.mean(acc)),
                         accuracy_std=float(np.std(acc, ddof=1)) if len(acc) > 1 else "",
                         macro_f1_mean=float(np.mean(f1)),
                         macro_f1_std=float(np.std(f1, ddof=1)) if len(f1) > 1 else ""))
    return rows


# ============================================================================================
# Run records
# ============================================================================================
def new_run_dir(root, method, seed, overwrite=False):
    """Dhan-Shomadhan/experiments/<method>/seed_<seed>/ (refuses to overwrite an existing run by default)."""
    if method not in METHODS and not method.startswith("resnet18_checks"):
        raise ValueError(f"method must be one of {METHODS}")
    directory = Path(root).resolve() / "experiments" / method / f"seed_{seed}"
    if directory.exists() and not overwrite:
        raise ValueError(f"{directory} already exists; choose another output or pass overwrite=True")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def base_run_config(method, seed, version, datasets, device, root=DATASET, audit=AUDIT, **extra):
    """Fields every CNN run_config.json needs; each variant adds its own settings through **extra."""
    return dict(method=method, seed=seed, model_seed=seed, manifest_version=manifest_version_name(version, audit),
                manifest_sha256={p: sha_file(Path(version) / f"seed_{seed}" / f"{p}.csv") for p in PARTITIONS},
                common_code_sha256=sha_file(Path(__file__)), transforms=transform_settings(datasets),
                weights=weights_record(root), batch_size=BATCH_SIZE, selection_rule=SELECTION_RULE,
                batchnorm="running statistics fixed at ImageNet values (kept in eval mode during training)",
                class_mapping=CLASSES, background_mapping=BACKGROUNDS, environment=environment(device), **extra)


def write_history(path, rows):
    write_csv(path, rows)


__all__ = [n for n in dir() if not n.startswith("_")]
