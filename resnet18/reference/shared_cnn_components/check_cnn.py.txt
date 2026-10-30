"""Implementation checks and small validation runs for the shared ResNet-18 components (Role 3).

Run from the repository root:
    python -m resnet18.check_cnn download-weights    # ImageNet ResNet-18 weights -> Dhan-Shomadhan/pretrained/
    python -m resnet18.check_cnn check               # loader, transforms, weights, freezing, BatchNorm, export
    python -m resnet18.check_cnn preview             # figure of evaluation inputs and training augmentations
    python -m resnet18.check_cnn smoke               # short head-only run on seed 11 (validation only)
    python -m resnet18.check_cnn compare-resize      # portrait vs squash vs pad on seed 11 validation data

Nothing here reads test predictions into a decision: smoke and compare-resize use training and validation
images only. Outputs go to Dhan-Shomadhan/experiments/resnet18_checks/, separate from the reported runs.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_preprocessing.prepare_dataset import CLASSES, SEEDS, write_csv, write_json
from resnet18 import common as cc

CHECK_DIR = "resnet18_checks"
EXPECTED_SIZES = {11: (777, 164, 165), 22: (777, 165, 164), 33: (777, 164, 165), 44: (777, 165, 164), 55: (777, 163, 166)}
CLASS_NAMES = sorted(CLASSES, key=CLASSES.get)


def _status(ok, text):
    print(f"   [{'PASS' if ok else 'FAIL'}] {text}")
    return ok


def check(root, audit, seed=11, verify_images=True, workers=0):
    root, ok = Path(root), True
    print("1. Dataset files")
    if verify_images:
        from data_preprocessing.verify_dataset import verify
        try:
            verify(root, audit)
            ok &= _status(True, "verify_dataset: images, metadata and manifests match the audit")
        except SystemExit as e:
            ok &= _status(False, f"verify_dataset failed: {e}")
            return False
    else:
        print("   (skipped image hashing; run python data_preprocessing/verify_dataset.py separately)")

    print("2. Shared splits through RiceDataset (current version, all seeds)")
    for s in SEEDS:
        data, version = cc.load_shared_splits(root, audit, s)
        sizes = tuple(len(data[p]) for p in ("train", "validation", "test"))
        ok &= _status(sizes == EXPECTED_SIZES[s], f"seed {s}: {sizes} from {cc.manifest_version_name(version, audit)}")
    data, version = cc.load_shared_splits(root, audit, seed)

    print("3. Transforms and batches")
    h, w = cc.RESIZE_MODES[cc.RESIZE_MODE]
    for part in ("train", "validation"):
        dl = cc.make_loader(data[part], train=part == "train", seed=seed, batch_size=8, workers=workers)
        t0 = time.time()
        batch = next(iter(dl))
        x = batch["image"]
        ok &= _status(tuple(x.shape) == (8, 3, h, w) and x.dtype == torch.float32 and bool(torch.isfinite(x).all()),
                      f"{part}: batch {tuple(x.shape)} {x.dtype}, range [{x.min():.2f}, {x.max():.2f}], "
                      f"{time.time() - t0:.1f}s for 8 images")
        ok &= _status(set(batch.keys()) == {"image", "disease_id", "background_id", "image_id"} and
                      int(batch["disease_id"].max()) < cc.NUM_CLASSES, f"{part}: fields and label range")
    ev = cc.EvalTransform()
    img = data["validation"].rows[0]
    from data_preprocessing.prepare_dataset import rgb_image
    pil = rgb_image(cc.source_path(root, img["path"]))
    ok &= _status(torch.equal(ev(pil), ev(pil)), "evaluation transform is deterministic")
    a1 = cc.TrainTransform(seed)(pil)
    a2 = cc.TrainTransform(seed)(pil)
    ok &= _status(torch.equal(a1, a2), "training augmentation repeats exactly for the same seed")
    ok &= _status(not torch.equal(a1, cc.TrainTransform(seed + 1)(pil)),
                  "different seeds give different augmentations")

    print("4. Pretrained weights")
    model = cc.build_resnet18(root)
    ok &= _status(True, f"{cc.weights_path(root)} verified ({cc.WEIGHTS_SHA256_PREFIX}...), every backbone tensor loaded")
    model.eval()
    with torch.no_grad():
        out = model(batch["image"])
    ok &= _status(tuple(out.shape) == (8, cc.NUM_CLASSES), f"forward pass gives {tuple(out.shape)}")

    print("5. Freezing")
    cc.set_trainable(model, {"fc": 1e-3})
    n_head = cc.count_parameters(model)
    cc.set_trainable(model, {"layer4": 1e-5, "fc": 1e-4})
    n_l4 = cc.count_parameters(model)
    ok &= _status(n_head["trainable"] == 512 * cc.NUM_CLASSES + cc.NUM_CLASSES,
                  f"frozen variant trains {n_head['trainable']:,} of {n_head['total']:,} parameters")
    ok &= _status(n_l4["trainable"] > n_head["trainable"], f"fine-tuned variant trains {n_l4['trainable']:,}")

    print("6. One training step (fine-tuned setting): BatchNorm statistics and frozen weights unchanged")
    groups = cc.set_trainable(model, {"layer4": 1e-5, "fc": 1e-4})
    opt = torch.optim.Adam(groups)
    snap = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    bn_modes = {m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)}
    loss = F.cross_entropy(model(batch["image"]), batch["disease_id"])
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = model.state_dict()
    running = [k for k in snap if "running_" in k]
    ok &= _status(bn_modes == {False}, "all BatchNorm layers stay in eval mode after model.train()")
    ok &= _status(all(torch.equal(snap[k], after[k]) for k in running), f"{len(running)} running-statistic tensors unchanged")
    frozen_keys = [k for k in snap if k.startswith(("conv1", "bn1", "layer1", "layer2", "layer3")) and "running_" not in k
                   and "num_batches" not in k]
    ok &= _status(all(torch.equal(snap[k], after[k]) for k in frozen_keys), "stem and layer1-3 weights unchanged")
    ok &= _status(not torch.equal(snap["fc.weight"], after["fc.weight"]) and
                  not torch.equal(snap["layer4.1.conv2.weight"], after["layer4.1.conv2.weight"]), "layer4 and head updated")

    print("7. Prediction export round trip (8 validation images, untrained head)")
    sub = cc.load_shared_splits(root, audit, seed)[0]["validation"]
    sub.rows = sub.rows[:8]
    res = cc.predict(model, cc.make_loader(sub, False, seed, batch_size=4), "cpu")
    ok &= _status(cc.check_predictions(res["rows"], sub), "every image predicted once, labels match manifest")
    with tempfile.TemporaryDirectory() as tmp:
        cc.write_predictions(Path(tmp) / "p.csv", res["rows"], seed, "frozen_resnet18")
        header = (Path(tmp) / "p.csv").read_text().splitlines()[0]
    expected = "image_id,seed,method,disease_id,predicted_disease_id,background_id," + ",".join(f"prob_{k}" for k in range(5))
    ok &= _status(header == expected, f"columns: {header}")
    with tempfile.TemporaryDirectory() as tmp:
        p = cc.save_checkpoint(Path(tmp) / "c.pt", model, method="check", seed=seed)
        m2, meta = cc.load_checkpoint(p)
        ok &= _status(all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), m2.state_dict().values())),
                      "checkpoint save/load restores identical weights")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - see above")
    return ok


def preview(root, audit, seed=11, n=4, out=None):
    """Evaluation input for each resize mode and four training augmentations, field and white images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data_preprocessing.prepare_dataset import rgb_image
    data, _ = cc.load_shared_splits(root, audit, seed)
    rows = data["train"].rows
    rng = np.random.default_rng(seed)
    pick = []
    for bg in ("0", "1"):
        cand = [r for r in rows if r["background_id"] == bg]
        pick += [cand[i] for i in rng.choice(len(cand), n // 2, replace=False)]
    modes = list(cc.RESIZE_MODES)
    cols = len(modes) + 4
    fig, axes = plt.subplots(len(pick), cols, figsize=(1.9 * cols, 3.0 * len(pick)))
    aug = cc.TrainTransform(seed)
    for i, r in enumerate(pick):
        pil = rgb_image(cc.source_path(root, r["path"]))
        for j, m in enumerate(modes):
            axes[i, j].imshow(cc.EvalTransform(m).pil(pil))
            axes[i, j].set_title(f"eval: {m}\n{cc.RESIZE_MODES[m][0]}x{cc.RESIZE_MODES[m][1]}", fontsize=7)
        for j in range(4):
            axes[i, len(modes) + j].imshow(aug.pil(pil))
            axes[i, len(modes) + j].set_title(f"train aug ({cc.RESIZE_MODE})", fontsize=7)
        axes[i, 0].set_ylabel(f"{CLASS_NAMES[int(r['disease_id'])]}\n{r['background']}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    out = Path(out or Path(root) / "experiments" / CHECK_DIR / "figures" / f"transform_preview_seed_{seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")


def _head_only_run(root, audit, seed, epochs, resize_mode, device, workers, limit=None, log_prefix=""):
    """Frozen-backbone training on train, selection on validation. Returns (tracker, history)."""
    cc.set_seed(seed)
    data, version = cc.load_shared_splits(root, audit, seed, resize_mode=resize_mode)
    if limit:
        for part in ("train", "validation"):
            data[part].rows = data[part].rows[:limit]
    model = cc.build_resnet18(root).to(device)
    opt = torch.optim.Adam(cc.set_trainable(model, {"fc": 1e-3}))
    tr = cc.make_loader(data["train"], True, seed, workers=workers)
    va = cc.make_loader(data["validation"], False, seed, workers=workers)
    tracker, history = cc.BestTracker(), []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for batch in tr:
            x, y = batch["image"].to(device), batch["disease_id"].to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(y), n + len(y)
        val = cc.predict(model, va, device)
        cc.check_predictions(val["rows"], data["validation"])
        best = tracker.update(val["macro_f1"], val["loss"], epoch)
        history.append(dict(epoch=epoch, train_loss=total / n, validation_loss=val["loss"],
                            validation_accuracy=val["accuracy"], validation_macro_f1=val["macro_f1"],
                            selected=int(best), seconds=round(time.time() - t0, 1)))
        print(f"   {log_prefix}epoch {epoch:2d}  train_loss {total / n:.4f}  val_loss {val['loss']:.4f}  "
              f"val_acc {val['accuracy']:.4f}  val_macroF1 {val['macro_f1']:.4f}  {history[-1]['seconds']}s"
              f"{'  *' if best else ''}")
        if tracker.should_stop:
            break
    return tracker, history, data, version


def smoke(root, audit, seed=11, epochs=3, limit=None, force_cpu=False, workers=0):
    device = cc.get_device(force_cpu)
    print(f"Smoke run: frozen backbone, head only, seed {seed}, {epochs} epochs, device {device}")
    tracker, history, data, version = _head_only_run(root, audit, seed, epochs, cc.RESIZE_MODE, device, workers, limit)
    out = cc.new_run_dir(root, f"{CHECK_DIR}/smoke", seed, overwrite=True)
    write_csv(out / "validation_history.csv", history)
    write_json(out / "run_config.json", cc.base_run_config(
        "resnet18_checks/smoke", seed, version, data, device, root, audit, epochs=epochs, limit=limit,
        trainable={"fc": 1e-3}, note="implementation check only; not a reported experiment; test set not used"))
    falling = history[-1]["train_loss"] < history[0]["train_loss"]
    print(f"Training loss {'decreased' if falling else 'did NOT decrease'}; best validation macro-F1 "
          f"{tracker.best_f1:.4f} at epoch {tracker.best_epoch}. Files in {out}")


def compare_resize(root, audit, seed=11, epochs=10, force_cpu=False, workers=0):
    """Chooses the input-size policy on validation data only (frozen backbone, head-only training)."""
    device = cc.get_device(force_cpu)
    rows = []
    for mode in cc.RESIZE_MODES:
        print(f"Resize mode '{mode}' {cc.RESIZE_MODES[mode]}:")
        tracker, history, _, _ = _head_only_run(root, audit, seed, epochs, mode, device, workers, log_prefix=f"{mode:8s} ")
        rows.append(dict(resize_mode=mode, input_hw="x".join(map(str, cc.RESIZE_MODES[mode])), seed=seed, epochs_run=len(history),
                         best_epoch=tracker.best_epoch, best_validation_macro_f1=tracker.best_f1,
                         validation_loss_at_best=tracker.best_loss,
                         seconds_per_epoch=round(float(np.mean([h["seconds"] for h in history])), 1)))
    out = Path(root) / "experiments" / CHECK_DIR / f"compare_resize_seed_{seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows)
    print("\nValidation macro-F1 by resize mode (test images not used):")
    for r in rows:
        print(f"   {r['resize_mode']:8s} {r['input_hw']:8s} best {r['best_validation_macro_f1']:.4f} (epoch {r['best_epoch']}), "
              f"{r['seconds_per_epoch']}s/epoch")
    print(f"Saved {out}. Agree the mode with Roles 5 and 6, then set RESIZE_MODE in resnet18/common.py.")


def main():
    # Shared options work before or after the command name (e.g. "smoke --workers 4" or "--workers 4 smoke").
    def shared(parser, top):
        d = (lambda v: v) if top else (lambda v: argparse.SUPPRESS)
        parser.add_argument("--dataset", type=Path, default=d(cc.DATASET))
        parser.add_argument("--audit", type=Path, default=d(cc.AUDIT))
        parser.add_argument("--seed", type=int, default=d(11), choices=SEEDS)
        parser.add_argument("--workers", type=int, default=d(0), help="DataLoader worker processes (faster JPEG decoding)")
        parser.add_argument("--cpu", action="store_true", default=d(False))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    shared(ap, True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    shared(sub.add_parser("download-weights"), False)
    c = sub.add_parser("check")
    shared(c, False)
    c.add_argument("--skip-verify", action="store_true", help="skip hashing every source image")
    shared(sub.add_parser("preview"), False)
    s = sub.add_parser("smoke")
    shared(s, False)
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--limit", type=int, help="use only the first N train/validation images")
    r = sub.add_parser("compare-resize")
    shared(r, False)
    r.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "download-weights":
        cc.download_weights(a.dataset)
    elif a.cmd == "check":
        raise SystemExit(0 if check(a.dataset, a.audit, a.seed, not a.skip_verify, a.workers) else 1)
    elif a.cmd == "preview":
        preview(a.dataset, a.audit, a.seed)
    elif a.cmd == "smoke":
        smoke(a.dataset, a.audit, a.seed, a.epochs, a.limit, a.cpu, a.workers)
    elif a.cmd == "compare-resize":
        compare_resize(a.dataset, a.audit, a.seed, a.epochs, a.cpu, a.workers)


if __name__ == "__main__":
    main()
