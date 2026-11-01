"""Train frozen and partially fine-tuned ResNet-18 models on the shared splits.

Completed local stages can be reused when checkpoints are present. Supply
--output with a fresh directory to train separately from the recorded results.
Run: python -u -m resnet18.run_role6 --output Dhan-Shomadhan/experiments/new-run
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from resnet18 import common as cc, finetune, train_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dataset", type=Path, default=cc.DATASET)
    parser.add_argument("--audit", type=Path, default=cc.AUDIT)
    parser.add_argument("--output", type=Path, help="Experiment output root; use a fresh directory for retraining")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--seeds", type=int, nargs="+", choices=cc.SEEDS, default=list(cc.SEEDS))
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Invalid workers, threads, or duplicate seeds")
    output = (args.output or args.dataset / "experiments").resolve()
    # A committed result bundle is sufficient for scoring, but cannot resume
    # training without its checkpoints. Reject it before starting any seed.
    for seed in args.seeds:
        frozen = output / "frozen_resnet18" / f"seed_{seed}"
        fine = output / finetune.METHOD / f"seed_{seed}"
        if frozen.exists() and not all((frozen / name).is_file() for name in
                ("run_config.json", "best_checkpoint.pt", "metrics.json", "test_predictions.csv")):
            parser.error(f"Cannot resume {frozen}: training checkpoints or outputs are missing. Use --output with a fresh directory.")
        if fine.exists():
            config_path = fine / "run_config.json"
            if not config_path.is_file():
                parser.error(f"Incomplete training directory: {fine}. Use a fresh --output directory.")
            config = json.loads(config_path.read_text())
            selected = fine / config.get("selected_checkpoint", "missing")
            if not selected.is_file():
                parser.error(f"Selected checkpoint missing: {selected}. Use a fresh --output directory.")
            if config.get("status") not in ("selected_validation_only", "test_evaluated"):
                parser.error(f"Unrecognised training status in {config_path}")
            if config["status"] == "test_evaluated" and not all((fine / name).is_file() for name in
                    ("metrics.json", "test_predictions.csv")):
                parser.error(f"Incomplete test outputs in {fine}")
    torch.set_num_threads(args.threads)
    device = cc.get_device(args.cpu)
    print(f"Training device: {device}; GPU: {torch.cuda.get_device_name() if device.type == 'cuda' else 'none'}", flush=True)
    frozen_args = SimpleNamespace(dataset=args.dataset, audit=args.audit, output=output, resize_mode=cc.RESIZE_MODE,
                                  batch_size=32, workers=args.workers, overwrite=False,
                                  learning_rate=1e-3, patience=5, epochs=30)
    fine_args = SimpleNamespace(root=args.dataset, audit=args.audit, output=output, workers=args.workers,
                                cpu=args.cpu, epochs=20, checkpoint_name="best_checkpoint.pt",
                                frozen_dir=output / "frozen_resnet18")
    for seed in args.seeds:
        out = output / "frozen_resnet18" / f"seed_{seed}"
        if not (out / "run_config.json").is_file():
            print(f"Generating missing same-seed frozen prerequisite: {seed}", flush=True)
            train_frozen.run_seed(seed, frozen_args, device)
        else:
            print(f"Reusing completed frozen seed {seed}", flush=True)
    for seed in args.seeds:
        out = output / finetune.METHOD / f"seed_{seed}"
        if not (out / "run_config.json").is_file():
            finetune.select_seed(fine_args, seed)
        else:
            print(f"Reusing selected fine-tuned seed {seed}", flush=True)
    # All configuration/checkpoint choices are now fixed, before any fine-tuned test evaluation.
    scores = []
    for seed in args.seeds:
        out = output / finetune.METHOD / f"seed_{seed}"
        config = json.loads((out / "run_config.json").read_text())
        if config["status"] == "test_evaluated":
            scores.append(json.loads((out / "metrics.json").read_text()))
        else:
            scores.append(finetune.evaluate_seed(fine_args, seed))
        print(f"Completed fine-tuned test evaluation for seed {seed}", flush=True)
    label = "_".join(map(str, args.seeds))
    cc.write_csv(output / finetune.METHOD / f"summary_seeds_{label}.csv",
                 cc.summary_rows(scores))
    print(f"All requested CNN runs completed. Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
