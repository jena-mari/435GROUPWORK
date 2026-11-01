"""Run prerequisite frozen models, Person 6 selection, then test evaluation.

Uses Person 5's supplied train_frozen.py unchanged. Existing completed stages
are reused. Incomplete output folders are preserved and require investigation.
Run: python -u -m resnet18.run_role6 --workers 4
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
    parser.add_argument("--seeds", type=int, nargs="+", choices=cc.SEEDS, default=list(cc.SEEDS))
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Invalid workers, threads, or duplicate seeds")
    torch.set_num_threads(args.threads)
    device = cc.get_device()
    print(f"Training device: {device}; GPU: {torch.cuda.get_device_name() if device.type == 'cuda' else 'none'}", flush=True)
    frozen_args = SimpleNamespace(dataset=cc.DATASET, audit=cc.AUDIT, resize_mode=cc.RESIZE_MODE,
                                  batch_size=32, workers=args.workers, overwrite=False,
                                  learning_rate=1e-3, patience=5, epochs=30)
    fine_args = SimpleNamespace(root=cc.DATASET, audit=cc.AUDIT, workers=args.workers,
                                cpu=False, epochs=20, checkpoint_name="best_checkpoint.pt",
                                frozen_dir=cc.DATASET / "experiments" / "frozen_resnet18")
    for seed in args.seeds:
        out = frozen_args.dataset / "experiments" / "frozen_resnet18" / f"seed_{seed}"
        if not (out / "run_config.json").is_file():
            print(f"Generating missing same-seed frozen prerequisite: {seed}", flush=True)
            train_frozen.run_seed(seed, frozen_args, device)
        else:
            print(f"Reusing completed frozen seed {seed}", flush=True)
    for seed in args.seeds:
        out = cc.DATASET / "experiments" / finetune.METHOD / f"seed_{seed}"
        if not (out / "run_config.json").is_file():
            finetune.select_seed(fine_args, seed)
        else:
            print(f"Reusing selected fine-tuned seed {seed}", flush=True)
    # All configuration/checkpoint choices are now fixed, before any fine-tuned test evaluation.
    scores = []
    for seed in args.seeds:
        out = cc.DATASET / "experiments" / finetune.METHOD / f"seed_{seed}"
        config = json.loads((out / "run_config.json").read_text())
        if config["status"] == "test_evaluated":
            scores.append(json.loads((out / "metrics.json").read_text()))
        else:
            scores.append(finetune.evaluate_seed(fine_args, seed))
        print(f"Completed fine-tuned test evaluation for seed {seed}", flush=True)
    label = "_".join(map(str, args.seeds))
    cc.write_csv(cc.DATASET / "experiments" / finetune.METHOD / f"summary_seeds_{label}.csv",
                 cc.summary_rows(scores))
    print("All requested Role 6 runs completed.", flush=True)


if __name__ == "__main__":
    main()
