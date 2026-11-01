"""Train the frozen-backbone ResNet-18 experiment (Person 5, CSCI435).

Run from the repository root:
    python -m resnet18.train_frozen

The script uses the project's current shared manifests and never uses test data
for optimisation, hyperparameter selection, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from resnet18 import common as cc

METHOD = "frozen_resnet18"
DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def train_epoch(model, loader, optimiser, device):
    model.train()  # ResNet18.train() keeps BatchNorm running statistics frozen.
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["disease_id"].to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimiser.step()
        n = labels.size(0)
        total_loss += loss.item() * n
        total += n
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return {"train_loss": total_loss / max(total, 1), "train_accuracy": correct / max(total, 1)}


def _json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_seed(seed, args, device):
    cc.set_seed(seed)
    dataset_root = Path(args.dataset)
    audit_root = Path(args.audit) if args.audit else cc.AUDIT
    datasets, version = cc.load_shared_splits(
        root=dataset_root, audit=audit_root, seed=seed, resize_mode=args.resize_mode
    )
    train_loader = cc.make_loader(datasets["train"], train=True, seed=seed,
                                  batch_size=args.batch_size, workers=args.workers)
    val_loader = cc.make_loader(datasets["validation"], train=False, seed=seed,
                                batch_size=args.batch_size, workers=args.workers)
    test_loader = cc.make_loader(datasets["test"], train=False, seed=seed,
                                 batch_size=args.batch_size, workers=args.workers)

    run_dir = cc.new_run_dir(dataset_root, METHOD, seed, overwrite=args.overwrite,
                             output_root=getattr(args, "output", None))
    model = cc.build_resnet18(dataset_root, pretrained=True).to(device)
    parameter_groups = cc.set_trainable(model, {"fc": args.learning_rate})
    params = cc.count_parameters(model)
    if params["trainable"] != sum(p.numel() for p in model.fc.parameters()):
        raise RuntimeError("Frozen experiment has trainable backbone parameters")
    optimiser = torch.optim.Adam(parameter_groups)
    tracker = cc.BestTracker(patience=args.patience)
    history = []
    checkpoint_path = run_dir / "best_checkpoint.pt"
    start = time.time()

    print(f"\n[{METHOD}] seed={seed}, device={device}, trainable={params['trainable']:,}/{params['total']:,}")
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimiser, device)
        val_stats = cc.predict(model, val_loader, device)
        row = {"epoch": epoch, **train_stats,
               "validation_loss": val_stats["loss"],
               "validation_accuracy": val_stats["accuracy"],
               "validation_macro_f1": val_stats["macro_f1"]}
        history.append(row)
        improved = tracker.update(val_stats["macro_f1"], val_stats["loss"], epoch)
        if improved:
            cc.save_checkpoint(checkpoint_path, model, method=METHOD, seed=seed, epoch=epoch,
                               validation_macro_f1=val_stats["macro_f1"],
                               validation_loss=val_stats["loss"], learning_rate=args.learning_rate,
                               resize_mode=args.resize_mode, trainable_parameters=params["trainable"])
        print(f"seed {seed} epoch {epoch:02d}/{args.epochs}: "
              f"train_loss={train_stats['train_loss']:.4f} "
              f"val_macro_f1={val_stats['macro_f1']:.4f} "
              f"{'[best]' if improved else ''}")
        if tracker.should_stop:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    cc.write_history(run_dir / "training_history.csv", history)
    model, checkpoint_meta = cc.load_checkpoint(checkpoint_path, model=model, device=device)
    test_stats = cc.predict(model, test_loader, device)
    cc.check_predictions(test_stats["rows"], datasets["test"])
    cc.write_predictions(run_dir / "test_predictions.csv", test_stats["rows"], seed, METHOD)
    metrics = cc.get_scores(test_stats["rows"])
    _json_write(run_dir / "metrics.json", {
        "method": METHOD, "seed": seed, "best_epoch": tracker.best_epoch,
        "best_validation_macro_f1": tracker.best_f1,
        "best_validation_loss": tracker.best_loss,
        "test_loss": test_stats["loss"], "test_accuracy": test_stats["accuracy"],
        "test_macro_f1": test_stats["macro_f1"], "scenarios": metrics,
        "elapsed_seconds": round(time.time() - start, 2),
    })
    config = cc.base_run_config(
        METHOD, seed, version, datasets, device, root=dataset_root, audit=audit_root,
        epochs_requested=args.epochs, epochs_completed=len(history), learning_rate=args.learning_rate,
        optimiser="Adam", trainable_stages={"fc": args.learning_rate},
        parameter_counts=params, patience=args.patience, resize_mode=args.resize_mode,
        checkpoint=str(checkpoint_path), best_epoch=tracker.best_epoch,
        best_validation_macro_f1=tracker.best_f1, workers=args.workers,
    )
    _json_write(run_dir / "run_config.json", config)
    print(f"seed {seed} TEST: accuracy={test_stats['accuracy']:.4f}, "
          f"macro-F1={test_stats['macro_f1']:.4f}; outputs: {run_dir}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(cc.DATASET), help="dataset root containing the source image folders")
    parser.add_argument("--audit", default=None, help="shared preprocessing audit directory (defaults to project audit)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), choices=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=cc.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=cc.PATIENCE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resize-mode", choices=tuple(cc.RESIZE_MODES), default=cc.RESIZE_MODE)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if an accelerator is available")
    parser.add_argument("--overwrite", action="store_true", help="replace outputs for selected seeds")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.patience < 1:
        parser.error("epochs, batch-size, patience must be positive and learning-rate must be > 0")

    device = cc.get_device(force_cpu=args.cpu)
    results = []
    for seed in args.seeds:
        results.append(run_seed(seed, args, device))

    summary = cc.summary_rows(results)
    summary_path = Path(args.dataset).resolve() / "experiments" / METHOD / "summary.csv"
    cc.write_history(summary_path, summary)
    print("\nMean and sample standard deviation across completed runs:")
    for row in summary:
        print(f"{row['scenario']:>5}: accuracy={row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}; "
              f"macro-F1={row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
