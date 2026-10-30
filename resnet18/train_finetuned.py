"""Run the same-seed partially fine-tuned ResNet-18 experiment.

The driver uses the shared CNN contracts. It starts each run from the selected
frozen checkpoint for that seed, trains layer4 and the five-class head, and
keeps test images out of validation selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data_preprocessing.prepare_dataset import SEEDS, sha_file, write_csv, write_json
from data_preprocessing.verify_dataset import verify
from . import common as cc

METHOD = "finetuned_resnet18"
AUDIT = Path(__file__).resolve().parents[1] / "Dhan-Shomadhan/preprocessing"


def run_seed(seed, args, device):
    cc.seed_everything(seed)
    root = args.dataset.resolve()
    data = cc.make_shared_data(root, args.audit, seed, batch_size=args.batch_size,
                               num_workers=args.workers)
    directory = args.output.resolve() / f"seed_{seed}"
    frozen_checkpoint = args.frozen / f"seed_{seed}/best.pt"
    if not frozen_checkpoint.is_file():
        raise FileNotFoundError(f"Missing selected frozen checkpoint: {frozen_checkpoint}")
    checkpoint = directory / "best.pt"

    if args.evaluate_only:
        config = json.loads((directory / "run_config.json").read_text())
        if config.get("context") != data.context or not config.get("training_complete"):
            raise ValueError("Evaluation requires a completed fine-tuning run with the same context")
        model = cc.build_model("finetuned", pretrained=False).to(device)
        payload = cc.load_checkpoint(checkpoint, model, data.context, "finetuned")
        if sha_file(checkpoint) != config.get("checkpoint_sha256") or payload["epoch"] != config.get("best_epoch"):
            raise ValueError("Selected fine-tuned checkpoint differs from the run record")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        model, frozen_metadata = cc.initialise_finetuned(frozen_checkpoint, data.context, device)
        optimizer = cc.make_optimizer(model, "finetuned")
        config = cc.run_configuration(model, optimizer, data, "finetuned", device)
        config.update(
            actual_batch_size=args.batch_size,
            actual_max_epochs=args.epochs,
            actual_patience=args.patience,
            workers=args.workers,
            dataset_root=str(root),
            audit_directory=str(args.audit.resolve()),
            frozen_checkpoint=str(frozen_checkpoint),
            frozen_checkpoint_sha256=sha_file(frozen_checkpoint),
            frozen_checkpoint_epoch=frozen_metadata["epoch"],
            driver_code_sha256=sha_file(Path(__file__)),
            checkpoint="best.pt",
            selection_rule="strictly highest validation macro-F1; ties count as no improvement",
            validation_only=args.validation_only,
            training_complete=False,
        )
        write_json(directory / "run_config.json", config)
        selector = cc.BestCheckpoint(checkpoint, data.context, "finetuned", args.patience)
        history, started = [], time.monotonic()
        for epoch in range(1, args.epochs + 1):
            train = cc.train_one_epoch(model, data.loaders["train"], optimizer, device, "finetuned")
            val = cc.evaluate(model, data.loaders["validation"], device)
            history.append(dict(
                epoch=epoch,
                training_loss=train["loss"],
                training_accuracy=train["accuracy"],
                validation_loss=val["loss"],
                validation_accuracy=val["accuracy"],
                validation_macro_f1=val["macro_f1"],
            ))
            stop = selector.update(val["macro_f1"], epoch, model, optimizer, config)
            write_csv(directory / "validation_history.csv", history)
            print(
                f"seed {seed} epoch {epoch}: train loss {train['loss']:.4f}, "
                f"validation macro-F1 {val['macro_f1']:.4f}, best epoch {selector.best_epoch}",
                flush=True,
            )
            if stop:
                break
        config.update(
            training_complete=True,
            epochs_completed=len(history),
            best_epoch=selector.best_epoch,
            best_validation_macro_f1=selector.best_score,
            elapsed_training_seconds=time.monotonic() - started,
            checkpoint_sha256=sha_file(checkpoint),
        )
        write_json(directory / "run_config.json", config)
        cc.load_checkpoint(checkpoint, model, data.context, "finetuned")
        validation_rows = cc.export_predictions(
            model, data.loaders["validation"], directory / "validation_predictions.csv",
            data.context, "finetuned", device,
        )
        write_json(directory / "validation_metrics.json", cc.score_predictions(validation_rows))

    if args.validation_only:
        print(f"Validation only: {directory}; test images evaluated: 0", flush=True)
        return None
    rows = cc.export_predictions(
        model, data.loaders["test"], directory / "test_predictions.csv",
        data.context, "finetuned", device,
    )
    scores = cc.score_predictions(rows)
    write_json(directory / "metrics.json", scores)
    write_json(directory / "test_evaluation.json", dict(
        checkpoint_sha256=sha_file(checkpoint), evaluation_device=str(device),
        context=data.context, images=len(rows),
    ))
    print(f"Saved selected-checkpoint test predictions: {directory}", flush=True)
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Dhan-Shomadhan"))
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--frozen", type=Path, required=True,
                        help="directory containing seed_<seed>/best.pt frozen checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0:
        parser.error("Use distinct seeds and positive epoch, batch-size and patience values")
    if args.evaluate_only and args.validation_only:
        parser.error("--evaluate-only cannot be combined with --validation-only")
    if not args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error("Save outputs under the dataset directory")
    for seed in args.seeds:
        directory = args.output / f"seed_{seed}"
        if args.evaluate_only:
            if not (directory / "best.pt").is_file() or not (directory / "run_config.json").is_file():
                parser.error(f"No completed run found: {directory}")
            if any((directory / name).exists() for name in (
                "test_predictions.csv", "test_predictions.metadata.json", "metrics.json", "test_evaluation.json",
            )):
                parser.error(f"Test outputs already exist: {directory}")
        elif directory.exists():
            parser.error(f"Run directory exists; choose a new output: {directory}")
    verify(args.dataset, args.audit)
    device = cc.choose_device(args.device)
    results = {seed: run_seed(seed, args, device) for seed in args.seeds}
    if not args.validation_only and set(results) == set(SEEDS):
        write_csv(args.output / "summary.csv", cc.summarise_runs(results))


if __name__ == "__main__":
    main()
