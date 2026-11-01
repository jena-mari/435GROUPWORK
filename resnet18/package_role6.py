"""Verify completed experiments, make figures/report, and build group handover ZIPs."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from resnet18 import common as cc, finetune as ft

ROOT = cc.REPO
OUT = ROOT / "role6_submission"
RUNS = ROOT / cc.DATASET / "experiments" / ft.METHOD
NAMES = [k.replace("_", " ").title() for k in sorted(cc.CLASSES, key=cc.CLASSES.get)]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def verify_run(seed):
    folder = RUNS / f"seed_{seed}"
    cfg = ft.read_json(folder / "run_config.json")
    if cfg["status"] != "test_evaluated" or cfg["seed"] != seed:
        raise ValueError(f"Seed {seed} is incomplete")
    data, version = cc.load_shared_splits(ROOT / cc.DATASET, cc.AUDIT, seed)
    checkpoint = folder / cfg["selected_checkpoint"]
    if cc.sha_file(checkpoint) != cfg["selected_checkpoint_sha256"]:
        raise ValueError(f"Seed {seed} checkpoint changed")
    frozen_source = Path(cfg["frozen_checkpoint"])
    if cc.sha_file(frozen_source) != cfg["frozen_sha256"]:
        raise ValueError("Frozen source checkpoint changed")
    if cc.sha_file(frozen_source.parent / "run_config.json") != cfg["frozen_config_sha256"]:
        raise ValueError("Frozen source configuration changed")
    model, _ = cc.load_checkpoint(checkpoint)
    frozen_model, _ = cc.load_checkpoint(frozen_source)
    frozen_state = frozen_model.state_dict()
    for name, value in model.state_dict().items():
        if not name.startswith(("layer4.", "fc.")) or "running_" in name or "num_batches_tracked" in name:
            if not torch.equal(value, frozen_state[name]):
                raise ValueError(f"Frozen parameter/statistic changed: {name}")
    del model, frozen_model
    if cfg["training_code_sha256"] != cc.sha_file(Path(ft.__file__)):
        raise ValueError("Training code changed since the run")
    if cfg["common_code_sha256"] != cc.sha_file(Path(cc.__file__)):
        raise ValueError("Shared CNN code changed since the run")
    for partition in cc.PARTITIONS:
        if cc.sha_file(version / f"seed_{seed}" / f"{partition}.csv") != cfg["manifest_sha256"][partition]:
            raise ValueError("Shared manifest changed")
    rows = read_csv(folder / "test_predictions.csv")
    for r in rows:
        if int(r["seed"]) != seed or r["method"] != ft.METHOD:
            raise ValueError("Prediction method/seed mismatch")
        for k in ("disease_id", "predicted_disease_id", "background_id"):
            r[k] = int(r[k])
        probs = np.array([float(r[f"prob_{k}"]) for k in range(5)])
        if not (np.isfinite(probs).all() and np.all(probs >= 0) and np.all(probs <= 1)
                and abs(probs.sum() - 1) < 1e-5 and 0 <= r["predicted_disease_id"] < 5
                and probs[r["predicted_disease_id"]] == probs.max()):
            raise ValueError("Invalid exported probabilities")
    cc.check_predictions(rows, data["test"])
    scores = cc.get_scores(rows)
    recorded = ft.read_json(folder / "metrics.json")
    if scores != recorded:
        raise ValueError(f"Seed {seed}: exported predictions do not reproduce metrics")
    frozen = ft.read_json(ROOT / cc.DATASET / "experiments" / "frozen_resnet18" / f"seed_{seed}" / "metrics.json")
    frozen_rows = read_csv(frozen_source.parent / "test_predictions.csv")
    for row in frozen_rows:
        for key in ("disease_id", "predicted_disease_id", "background_id"):
            row[key] = int(row[key])
        if int(row["seed"]) != seed or row["method"] != "frozen_resnet18":
            raise ValueError("Frozen prediction seed/method mismatch")
    cc.check_predictions(frozen_rows, data["test"])
    if cc.get_scores(frozen_rows) != frozen["scenarios"]:
        raise ValueError("Frozen comparison metrics do not match their prediction CSV")
    return cfg, scores, frozen["scenarios"], rows


def figures(seed, config, scores):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    run = RUNS / f"seed_{seed}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for candidate in config["candidates"]:
        i = candidate["candidate"]
        hist = read_csv(run / f"candidate_{i}" / "validation_history.csv")
        epochs = [int(h["epoch"]) for h in hist]
        axes[0].plot(epochs, [float(h["validation_macro_f1"]) for h in hist], marker=".", label=f"Candidate {i}")
        axes[1].plot(epochs, [float(h["train_loss"]) for h in hist], label=f"{i}: train")
        axes[1].plot(epochs, [float(h["validation_loss"]) for h in hist], linestyle="--", label=f"{i}: validation")
    chosen = config["selected"]
    axes[0].scatter([chosen["best_epoch"]], [chosen["validation_macro_f1"]], marker="*", color="black", s=130, label="Selected")
    axes[0].set(title="Validation macro-F1", ylim=(0, 1), xlabel="Additional epoch")
    axes[1].set(title="Cross-entropy loss", xlabel="Additional epoch")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f"Fine-tuned ResNet-18 · seed {seed} · training/validation only")
    fig.savefig(folder / f"learning_curves_seed_{seed}.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for ax, scenario in zip(axes, ("field", "white", "all")):
        matrix = np.array(scores[scenario]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (y, x), value in np.ndenumerate(matrix):
            ax.text(x, y, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=NAMES, yticklabels=NAMES,
               xlabel="Predicted disease", ylabel="True disease",
               title=f"{scenario.title()} (n={scores[scenario]['images']})")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.suptitle(f"Selected fine-tuned ResNet-18 · seed {seed} · test predictions")
    fig.savefig(folder / f"confusion_matrices_seed_{seed}.png", dpi=160)
    plt.close(fig)
    canonical = RUNS / "figures"
    canonical.mkdir(exist_ok=True)
    for name in (f"learning_curves_seed_{seed}.png", f"confusion_matrices_seed_{seed}.png"):
        shutil.copy2(folder / name, canonical / name)


def main():
    OUT.mkdir(exist_ok=True)
    if not (OUT / "README.md").exists():
        shutil.copy2(ROOT / "resnet18" / "ROLE6_FINE_TUNED_EXPERIMENTS.md", OUT / "README.md")
    results, frozen_results, configs, per_seed, classes, differences, failures = [], [], [], [], [], [], []
    for seed in cc.SEEDS:
        config, scores, frozen, rows = verify_run(seed)
        configs.append(config)
        results.append(scores)
        frozen_results.append(frozen)
        figures(seed, config, scores)
        for scenario, score in scores.items():
            per_seed.append(dict(seed=seed, scenario=scenario, images=score["images"], accuracy=score["accuracy"],
                                 macro_f1=score["macro_f1"], selected_candidate=config["selected"]["candidate"],
                                 selected_epoch=config["selected"]["best_epoch"]))
            differences.append(dict(seed=seed, scenario=scenario,
                                    accuracy_difference=score["accuracy"] - frozen[scenario]["accuracy"],
                                    macro_f1_difference=score["macro_f1"] - frozen[scenario]["macro_f1"]))
            for i, value in enumerate(score["per_class_f1"]):
                classes.append(dict(seed=seed, scenario=scenario, disease_id=i, disease=NAMES[i], f1=value))
        failures.extend(dict(seed=seed, image_id=r["image_id"], background_id=r["background_id"],
                             disease_id=r["disease_id"], predicted_disease_id=r["predicted_disease_id"])
                        for r in rows if r["disease_id"] != r["predicted_disease_id"])
    summary = cc.summary_rows(results)
    cc.write_csv(OUT / "results_by_seed.csv", per_seed)
    cc.write_csv(OUT / "summary.csv", summary)
    cc.write_csv(OUT / "per_class_f1.csv", classes)
    cc.write_csv(OUT / "paired_difference_from_frozen.csv", differences)
    cc.write_csv(OUT / "misclassified_test_images.csv", failures,
                 fields=["seed", "image_id", "background_id", "disease_id", "predicted_disease_id"])
    frozen_summary = cc.summary_rows(frozen_results)
    cc.write_csv(OUT / "frozen_prerequisite_summary.csv", frozen_summary)
    all_result = next(r for r in summary if r["scenario"] == "all")
    delta = np.mean([r["macro_f1_difference"] for r in differences if r["scenario"] == "all"])
    class_means = np.mean([r["all"]["per_class_f1"] for r in results], axis=0)
    strongest, weakest = int(class_means.argmax()), int(class_means.argmin())
    confusion = np.sum([r["all"]["confusion_matrix"] for r in results], axis=0)
    np.fill_diagonal(confusion, 0)
    confused_true, confused_pred = np.unravel_index(confusion.argmax(), confusion.shape)
    field_mean = next(r for r in summary if r["scenario"] == "field")["macro_f1_mean"]
    white_mean = next(r for r in summary if r["scenario"] == "white")["macro_f1_mean"]
    lines = [
        "# Person 6 — Fine-tuned ResNet-18 experiments", "",
        "## Completion and contribution", "",
        "This submission implements the partly fine-tuned ResNet-18 workflow, selects settings using validation data, "
        "and provides test predictions, logs, checkpoints, figures, and interpretation for all five shared seeds. "
        "Person 5's supplied ZIP contained training code but no trained checkpoints or results. "
        "The prerequisite frozen checkpoints were therefore generated locally using the supplied `train_frozen.py` unchanged. "
        "These prerequisite results should be coordinated with Person 5; they are not claimed to have been supplied as completed runs.", "",
        "## Method and implementation", "",
        "Each run starts from the best frozen checkpoint for its matching seed. ImageNet-pretrained ResNet-18 uses a "
        "five-class linear classifier. Only `layer4` and `fc` are optimised; the stem and layers 1–3 remain fixed. "
        "All BatchNorm running statistics stay fixed, including during training; affine parameters in `layer4` may learn. "
        "The shared 448 × 224 portrait input, ImageNet normalisation, and training-only augmentation are reused unchanged. "
        "Background IDs are retained as evaluation metadata and are not model inputs or targets.", "",
        "Adam uses batch size 32 for at most 20 additional epochs per candidate. A predeclared validation search compares "
        "(`layer4`, `fc`) learning rates: candidate 0 uses `(1e-5, 1e-4)` and candidate 1 uses `(3e-5, 3e-4)`. Each candidate restarts from the same frozen "
        "checkpoint and model/augmentation seed. Highest validation macro-F1 selects the epoch and candidate; exact ties "
        "use lower validation loss, then the earlier candidate. Training stops after five consecutive epochs without improvement. "
        "All five fine-tuning selections finish before any fine-tuned test evaluation. No test metric is used to choose settings.", "",
        "The saved split version is `conservative_splits/bad260e9eda16df2`. Seeds 11, 22, 33, 44, and 55 are repeated "
        "random holdouts, not five-fold cross-validation. Every run has 777 training images; validation/test counts differ slightly. "
        "Configuration files record source-checkpoint hashes, manifest hashes, code hashes, trainable stages, learning rates, "
        "selected epoch, transforms, device, and package versions.", "",
        "## Test results", "",
        "| Seed | Images | Accuracy | Macro-F1 | Candidate | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        if row["scenario"] == "all":
            lines.append(f"| {row['seed']} | {row['images']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['selected_candidate']} | {row['selected_epoch']} |")
    lines += ["", "| Background subset | Accuracy, mean ± sample SD | Macro-F1, mean ± sample SD |",
              "|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['scenario']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} |")
    lines += ["", "## Interpretation", "",
              f"Across the five holdouts, combined test accuracy is {all_result['accuracy_mean']:.4f} ± {all_result['accuracy_std']:.4f} "
              f"and macro-F1 is {all_result['macro_f1_mean']:.4f} ± {all_result['macro_f1_std']:.4f}. "
              f"The mean paired macro-F1 difference from the locally generated frozen prerequisites is {delta:+.4f} "
              f"({delta * 100:+.2f} percentage points). "
              + ("Fine-tuning improved mean macro-F1 in these runs." if delta > 0 else "Fine-tuning did not improve mean macro-F1 in these runs."), "",
              f"Mean field-background macro-F1 is {field_mean:.4f}; mean white-background macro-F1 is {white_mean:.4f}. "
              f"The strongest disease by mean combined per-class F1 is {NAMES[strongest]} ({class_means[strongest]:.4f}), "
              f"and the weakest is {NAMES[weakest]} ({class_means[weakest]:.4f}). "
              f"The largest off-diagonal confusion pooled over the five test runs is {NAMES[confused_true]} predicted as "
              f"{NAMES[confused_pred]} ({confusion[confused_true, confused_pred]} prediction events). "
              "Those events may include the same image in more than one holdout.", "",
              "These are descriptive results, not evidence of statistical significance. Holdouts overlap, so their scores are "
              "not independent replicates. The same disease labels and exact test images must be used for comparisons with "
              "the SVM baseline and Person 5's final results. Combined macro-F1 is computed from combined predictions, "
              "rather than averaging field and white macro-F1.", "",
              "Field/white performance differences may reflect both image difficulty and different class/group composition; "
              "they do not establish a causal background effect. The conservative grouping policy assigns a 461-image "
              "white-background group to training in every run, limiting test coverage. Label agreement with metadata "
              "does not independently verify biological diagnoses. Learning curves can reveal divergence between training "
              "and validation, while confusion matrices and per-class F1 identify the particular diseases that remain difficult. "
              "The failure CSV lists all misclassified test images by seed for Person 4's case analysis.", "",
              "## Dataset attribution", "",
              "Md Fahad Hossain, Sheikh Abujar, Sheak Rashed Haider Noori and Syed Akhter Hossain (2021). "
              "Dhan-Shomadhan: A Dataset of Rice Leaf Disease Classification for Bangladeshi Local Rice. "
              "Mendeley Data, V1. DOI: `10.17632/znsxdctwtt.1`. The project's dataset documentation identifies "
              "the licence as Creative Commons Attribution 4.0 International. Original dataset images are not "
              "redistributed in this package.", "",
              "Architecture reference: Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun (2016). "
              "Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition. "
              "DOI: `10.1109/CVPR.2016.90`. "
              "[Official publication record](https://www.microsoft.com/en-us/research/publication/deep-residual-learning-for-image-recognition/).", "",
              "## Figures and files", "",
              "`figures/learning_curves_seed_*.png` shows both candidates and the selected validation checkpoint. "
              "`figures/confusion_matrices_seed_*.png` shows field, white, and combined test confusion matrices. "
              "`results/seed_*/` contains the selected configuration, candidate histories and validation predictions, "
              "test predictions and metrics. `per_class_f1.csv`, `paired_difference_from_frozen.csv`, and "
              "`misclassified_test_images.csv` support the group's evaluation section.", "",
              "## Verification", "",
              "All 1,106 original images, source metadata, and 15 manifests passed the audited checksum verifier. "
              "Submission generation checks each selected checkpoint hash, manifest hashes, code hash, test image IDs, "
              "true labels, background IDs, probability sums/ranges, and prediction classes. Metrics are independently "
              "recomputed from exported test CSVs and must match the saved metrics exactly. Automated tests verify "
              "that only `layer4`/`fc` change, BatchNorm statistics remain fixed, mismatched frozen sources are rejected, "
              "and validation selection never predicts the test partition.", ""]
    (OUT / "ROLE6_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    with plt.rc_context({"font.size": 11.5}):
        fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
        x = np.arange(len(cc.SEEDS))
        ax.bar(x - .18, [r["all"]["macro_f1"] for r in frozen_results], .36,
               label="Frozen prerequisite", color="#92A4AD")
        ax.bar(x + .18, [r["all"]["macro_f1"] for r in results], .36,
               label="Fine-tuned", color="#146C7A")
        ax.set(xticks=x, xticklabels=[str(s) for s in cc.SEEDS], xlabel="Holdout seed",
               ylabel="Combined test macro-F1", ylim=(0, 1))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", ncols=2, frameon=False, fontsize=10)
        comparison_figure = OUT / "figures" / "combined_macro_f1_by_seed.png"
        fig.savefig(comparison_figure, dpi=180)
        plt.close(fig)
        shutil.copy2(comparison_figure, RUNS / "figures" / comparison_figure.name)
    from resnet18.report_pdf import build_pdf
    pdf = build_pdf(OUT / "ROLE6_REPORT.md", ROOT / "output" / "pdf" / "Role6_Report.pdf", comparison_figure)
    shutil.copy2(pdf, OUT / "ROLE6_REPORT.pdf")
    for seed in cc.SEEDS:
        source = RUNS / f"seed_{seed}"
        dest = OUT / "results" / f"seed_{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        for file in source.rglob("*"):
            if file.is_file() and file.suffix in (".csv", ".json"):
                target = dest / file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
        frozen_source = ROOT / cc.DATASET / "experiments" / "frozen_resnet18" / f"seed_{seed}"
        frozen_dest = OUT / "frozen_prerequisite_results" / f"seed_{seed}"
        frozen_dest.mkdir(parents=True, exist_ok=True)
        for name in ("run_config.json", "training_history.csv", "test_predictions.csv", "metrics.json"):
            shutil.copy2(frozen_source / name, frozen_dest / name)
    # Include shared manifests and their provenance; original JPEGs are supplied separately.
    selected_sources = [ROOT / "resnet18", ROOT / "data_preprocessing"]
    source_dest = OUT / "source"
    for source in selected_sources:
        shutil.copytree(source, source_dest / source.name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    audit_dest = source_dest / cc.DATASET / "preprocessing"
    audit_dest.mkdir(parents=True, exist_ok=True)
    for path in cc.AUDIT.iterdir():
        if path.is_file() and (path.suffix in (".csv", ".json") or path.name in ("CURRENT_SPLIT_VERSION.txt", "VERIFICATION.txt")):
            shutil.copy2(path, audit_dest / path.name)
    shutil.copy2(ROOT / cc.DATASET / "Dhan-Shomadhan_picture_Information.csv", source_dest / cc.DATASET)
    shutil.copy2(ROOT / "run_role6.ps1", source_dest)
    # The report generator can be rerun from source/; copy the guide into its output location too.
    (source_dest / "role6_submission").mkdir(exist_ok=True)
    shutil.copy2(OUT / "README.md", source_dest / "role6_submission" / "README.md")
    version = cc.current_version()
    shutil.copytree(version, audit_dest / version.relative_to(cc.AUDIT), dirs_exist_ok=True)
    (OUT / "CHECKS_PASSED.txt").write_text(
        "All five experiments complete. Exported test predictions reproduce saved metrics.\n"
        "All IDs, labels, backgrounds, probabilities, code/manifest/checkpoint hashes verified.\n"
        "Dataset verification and automated test logs are in logs/.\n", encoding="utf-8")
    from resnet18.export_role6_layout import export_role5_layout
    export_role5_layout(ROOT, OUT)
    print("Verified results, report, and Role 5-style submission ZIPs are ready.")


if __name__ == "__main__":
    main()
