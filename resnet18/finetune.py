"""Person 6: validation selection followed by separate, locked test evaluation.

Run from the repository root: python -m resnet18.finetune --help
Uses the shared CNN implementation without changing the shared split manifests.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from resnet18 import common as cc

METHOD = "finetuned_resnet18"
# Predeclared small search; all candidates start from the identical frozen model.
CANDIDATES = ((1e-5, 1e-4), (3e-5, 3e-4))


def read_json(path):
    return json.loads(Path(path).read_text())


def transform_record(datasets):
    return json.loads(json.dumps(cc.transform_settings(datasets)))


def validate_frozen(config, seed, version, datasets, audit):
    """Require an attributable same-seed frozen model with matching inputs."""
    expected = {
        "method": "frozen_resnet18", "seed": seed,
        "manifest_version": cc.manifest_version_name(version, audit),
        "manifest_sha256": {p: cc.sha_file(version / f"seed_{seed}" / f"{p}.csv")
                            for p in cc.PARTITIONS},
        "transforms": transform_record(datasets),
        "trainable": {"fc": 1e-3},
    }
    for key in ("method", "seed", "manifest_version", "manifest_sha256", "transforms"):
        if config.get(key) != expected[key]:
            raise ValueError(f"Frozen run_config.json has incompatible {key}; use Person 5's matching run")
    stages = config.get("trainable_stages", config.get("trainable", {}))
    if set(stages) != {"fc"}:
        raise ValueError("Frozen source must record that only fc was trained")
    if config.get("batch_size") != cc.BATCH_SIZE:
        raise ValueError("Frozen source must use the shared batch size")
    if config.get("common_code_sha256") != cc.sha_file(Path(cc.__file__)):
        raise ValueError("Frozen source used a different shared CNN implementation")
    if config.get("batchnorm") != (
        "running statistics fixed at ImageNet values (kept in eval mode during training)"
    ):
        raise ValueError("Frozen source must record the shared fixed-BatchNorm policy")


def train_epoch(model, loader, optimizer, device):
    model.train()  # Shared model keeps BatchNorm running statistics fixed.
    total, correct, count = 0.0, 0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["disease_id"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; no checkpoint selected")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
    return total / count, correct / count


def select_seed(args, seed):
    device = cc.get_device(args.cpu)
    data, version = cc.load_shared_splits(args.root, args.audit, seed)
    frozen_dir = args.frozen_dir / f"seed_{seed}"
    source = frozen_dir / args.checkpoint_name
    if not source.is_file():
        raise FileNotFoundError(f"Person 5 checkpoint missing: {source}")
    frozen_config = read_json(frozen_dir / "run_config.json")
    validate_frozen(frozen_config, seed, version, data, args.audit)
    source_hash = cc.sha_file(source)
    _, meta = cc.load_checkpoint(source)
    if meta.get("seed") != seed or meta.get("method") != "frozen_resnet18":
        raise ValueError("Frozen checkpoint metadata must contain matching seed and method=frozen_resnet18")
    out = cc.new_run_dir(args.root, METHOD, seed)
    candidates = []
    for index, (layer_lr, head_lr) in enumerate(CANDIDATES):
        cc.set_seed(seed)
        # Fresh datasets also reset augmentation generators between candidates.
        data, version = cc.load_shared_splits(args.root, args.audit, seed)
        model, _ = cc.load_checkpoint(source, device=device)
        optimizer = torch.optim.Adam(cc.set_trainable(model, {"layer4": layer_lr, "fc": head_lr}))
        train = cc.make_loader(data["train"], True, seed, workers=args.workers)
        val = cc.make_loader(data["validation"], False, seed, workers=args.workers)
        tracker, history = cc.BestTracker(), []
        dest = out / f"candidate_{index}"
        dest.mkdir()
        for epoch in range(1, args.epochs + 1):
            started = time.monotonic()
            loss, accuracy = train_epoch(model, train, optimizer, device)
            result = cc.predict(model, val, device)
            cc.check_predictions(result["rows"], data["validation"])
            if not all(math.isfinite(result[k]) for k in ("loss", "macro_f1")):
                raise RuntimeError("Non-finite validation result")
            improved = tracker.update(result["macro_f1"], result["loss"], epoch)
            history.append(dict(epoch=epoch, train_loss=loss, train_accuracy=accuracy,
                                validation_loss=result["loss"], validation_accuracy=result["accuracy"],
                                validation_macro_f1=result["macro_f1"], selected=int(improved),
                                seconds=round(time.monotonic() - started, 2)))
            cc.write_history(dest / "validation_history.csv", history)
            if improved:
                cc.save_checkpoint(dest / "best.pt", model, method=METHOD, seed=seed, epoch=epoch,
                                   candidate=index, frozen_sha256=source_hash)
                cc.write_predictions(dest / "validation_predictions.csv", result["rows"], seed, METHOD)
            print(f"seed {seed} candidate {index} epoch {epoch}: val macro-F1={result['macro_f1']:.4f}", flush=True)
            if tracker.should_stop:
                break
        candidates.append(dict(candidate=index, layer4_lr=layer_lr, head_lr=head_lr,
                               best_epoch=tracker.best_epoch, validation_macro_f1=tracker.best_f1,
                               validation_loss=tracker.best_loss, epochs_run=len(history)))
        del train, val, optimizer, model
    # Exact ties across candidates keep the first predeclared candidate.
    best = max(candidates, key=lambda r: (r["validation_macro_f1"], -r["validation_loss"]))
    selected = out / f"candidate_{best['candidate']}" / "best.pt"
    config = cc.base_run_config(
        METHOD, seed, version, data, device, args.root, args.audit,
        trainable={"layer4": best["layer4_lr"], "fc": best["head_lr"]},
        optimizer="Adam", max_epochs=args.epochs, patience=cc.PATIENCE, workers=args.workers,
        frozen_checkpoint=str(source.resolve()), frozen_sha256=source_hash,
        frozen_config_sha256=cc.sha_file(frozen_dir / "run_config.json"),
        training_code_sha256=cc.sha_file(Path(__file__)), candidates=candidates,
        selected=best, selected_checkpoint=str(selected.relative_to(out)),
        selected_checkpoint_sha256=cc.sha_file(selected), status="selected_validation_only")
    # requires_grad is not saved in a state_dict; record the intended training mask explicitly.
    record_model, _ = cc.load_checkpoint(selected)
    cc.set_trainable(record_model, config["trainable"])
    config["parameter_counts"] = cc.count_parameters(record_model)
    cc.write_json(out / "run_config.json", config)
    cc.write_csv(out / "selection.csv", candidates)
    print(f"Selected seed {seed}: candidate {best['candidate']}, epoch {best['best_epoch']}. Test untouched.")


def evaluate_seed(args, seed):
    out = args.root.resolve() / "experiments" / METHOD / f"seed_{seed}"
    config = read_json(out / "run_config.json")
    if (out / "test_predictions.csv").exists() or (out / "metrics.json").exists():
        raise ValueError(f"Test outputs already exist in {out}; preserve the completed experiment")
    data, version = cc.load_shared_splits(args.root, args.audit, seed)
    for key, value in {
        "method": METHOD, "seed": seed, "status": "selected_validation_only",
        "manifest_version": cc.manifest_version_name(version, args.audit),
        "manifest_sha256": {p: cc.sha_file(version / f"seed_{seed}" / f"{p}.csv") for p in cc.PARTITIONS},
        "transforms": transform_record(data), "common_code_sha256": cc.sha_file(Path(cc.__file__)),
        "training_code_sha256": cc.sha_file(Path(__file__)),
    }.items():
        if config.get(key) != value:
            raise ValueError(f"Selected run has incompatible {key}; evaluation stopped")
    selected = (out / config["selected_checkpoint"]).resolve()
    if not selected.is_relative_to(out) or cc.sha_file(selected) != config["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint path or checksum changed")
    device = cc.get_device(args.cpu)
    model, meta = cc.load_checkpoint(selected, device=device)
    if meta.get("seed") != seed or meta.get("method") != METHOD:
        raise ValueError("Selected checkpoint metadata mismatch")
    result = cc.predict(model, cc.make_loader(data["test"], False, seed, workers=args.workers), device)
    cc.check_predictions(result["rows"], data["test"])
    scores = cc.get_scores(result["rows"])
    cc.write_predictions(out / "test_predictions.csv", result["rows"], seed, METHOD)
    cc.write_json(out / "metrics.json", scores)
    config.update(status="test_evaluated", evaluation_environment=cc.environment(device))
    cc.write_json(out / "run_config.json", config)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "evaluate"))
    parser.add_argument("--root", type=Path, default=cc.DATASET)
    parser.add_argument("--audit", type=Path, default=cc.AUDIT)
    parser.add_argument("--frozen-dir", type=Path, help="Folder containing Person 5's seed_*/ runs")
    parser.add_argument("--checkpoint-name", default="best_checkpoint.pt", help="Person 5's selected checkpoint filename")
    parser.add_argument("--seeds", nargs="+", type=int, choices=cc.SEEDS, default=list(cc.SEEDS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs must be positive, workers nonnegative, and seeds unique")
    args.frozen_dir = args.frozen_dir or args.root / "experiments" / "frozen_resnet18"
    try:
        if args.action == "select":
            # Check every prerequisite before creating any runs.
            for seed in args.seeds:
                for name in (args.checkpoint_name, "run_config.json"):
                    path = args.frozen_dir / f"seed_{seed}" / name
                    if not path.is_file():
                        raise FileNotFoundError(f"Need Person 5's same-seed frozen result: {path}")
            for seed in args.seeds:
                select_seed(args, seed)
        else:
            results = [evaluate_seed(args, seed) for seed in args.seeds]
            destination = args.root / "experiments" / METHOD
            label = "_".join(map(str, args.seeds))
            cc.write_csv(destination / f"summary_seeds_{label}.csv", cc.summary_rows(results))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Cannot continue: {exc}\n")


if __name__ == "__main__":
    main()
