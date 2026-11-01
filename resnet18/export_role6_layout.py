"""Package completed Role 6 artifacts using Person 5's project-root layout.

This only copies and reorganises completed artifacts. It never trains a model
or changes a split, checkpoint or prediction.
"""
import json
import shutil
import zipfile
from pathlib import Path


def export_role5_layout(repo, artifacts):
    repo, artifacts = Path(repo).resolve(), Path(artifacts).resolve()
    destination = repo / "CSCI435_Role6_FineTuned_ResNet18"
    marker = destination / "resnet18" / "ROLE6_LAYOUT_INFO.json"
    if destination.exists() and not marker.exists():
        raise ValueError(f"Refusing to overwrite an unrelated folder: {destination}")
    destination.mkdir(exist_ok=True)
    for name in ("resnet18", "data_preprocessing"):
        shutil.copytree(repo / name, destination / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    marker.write_text(json.dumps({"layout": "role5_project_root", "seeds": [11, 22, 33, 44, 55]}, indent=2))
    shutil.copytree(artifacts / "source" / "Dhan-Shomadhan", destination / "Dhan-Shomadhan", dirs_exist_ok=True)
    shutil.copy2(repo / ".gitignore", destination)
    shutil.copy2(repo / "run_role6.ps1", destination)
    method = destination / "Dhan-Shomadhan" / "experiments" / "finetuned_resnet18"
    method.mkdir(parents=True, exist_ok=True)
    for seed in (11, 22, 33, 44, 55):
        shutil.copytree(artifacts / "results" / f"seed_{seed}", method / f"seed_{seed}", dirs_exist_ok=True)
        frozen = destination / "Dhan-Shomadhan" / "experiments" / "frozen_resnet18" / f"seed_{seed}"
        shutil.copytree(artifacts / "frozen_prerequisite_results" / f"seed_{seed}", frozen, dirs_exist_ok=True)
    for name in ("figures", "logs"):
        shutil.copytree(artifacts / name, method / name, dirs_exist_ok=True)
    for file in artifacts.iterdir():
        if file.is_file() and file.suffix in (".csv", ".txt"):
            shutil.copy2(file, method / file.name)
    # Reports now live beside seed folders, figures and summary tables.
    report = (artifacts / "ROLE6_REPORT.md").read_text(encoding="utf-8")
    report = report.replace("`results/seed_*/`", "`seed_*/`")
    (method / "ROLE6_REPORT.md").write_text(report, encoding="utf-8")
    from resnet18.report_pdf import build_pdf
    build_pdf(method / "ROLE6_REPORT.md", method / "ROLE6_REPORT.pdf",
              method / "figures" / "combined_macro_f1_by_seed.png")
    readme = """# CSCI435 - Person 6: Fine-tuned ResNet-18

This archive uses the same project-root layout as Person 5's submission.
Code is directly in `resnet18/` and `data_preprocessing/`; manifests and results
are under `Dhan-Shomadhan/`. There is no `source/` wrapper directory.

Start with [the Role 6 guide](resnet18/ROLE6_FINE_TUNED_EXPERIMENTS.md).
The [report](Dhan-Shomadhan/experiments/finetuned_resnet18/ROLE6_REPORT.pdf)
and editable Markdown version are in the fine-tuned experiment folder.

```text
resnet18/                         # shared CNN, Role 5 and Role 6 code
data_preprocessing/               # shared dataset loader and verification
Dhan-Shomadhan/
  preprocessing/                 # saved shared manifests and audit
  experiments/
    finetuned_resnet18/
      seed_11/ ... seed_55/       # configurations, histories, predictions, metrics
      figures/                   # learning curves and confusion matrices
      logs/                      # execution and verification records
      ROLE6_REPORT.pdf
      ROLE6_REPORT.md
      summary.csv
    frozen_resnet18/             # locally generated prerequisite CSV/JSON results
GROUP_HANDOVER.md
run_role6.ps1
```

All five runs are complete. Mean combined test accuracy: **66.62%**;
macro-F1: **0.6317**. Person 5's code was used unchanged to generate the missing
frozen starting checkpoints. Details and limitations are in the report.

Extract `CSCI435_Role6_Selected_Checkpoints.zip` into this same root if model
weights are needed. It uses matching `Dhan-Shomadhan/experiments/...` paths.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    (destination / "GROUP_HANDOVER.md").write_text(
        "# Group handover - Person 6\n\n"
        "Read `resnet18/ROLE6_FINE_TUNED_EXPERIMENTS.md` for setup, file locations, "
        "class IDs and reproduction instructions.\n\n"
        "Person 4: use `Dhan-Shomadhan/experiments/finetuned_resnet18/seed_*/test_predictions.csv` "
        "and match on image_id and seed. Metrics, histories and figures are under the same method folder.\n\n"
        "Person 5: your supplied archive had code but no trained checkpoints. The frozen prerequisites "
        "were generated locally using your unchanged script; their CSV/JSON results are under "
        "`Dhan-Shomadhan/experiments/frozen_resnet18/`. Coordinate them with your final runs.\n\n"
        "The main ZIP contains code and result records. Selected fine-tuned model weights are "
        "in the separate matching-layout checkpoints ZIP. Original images are supplied separately.\n",
        encoding="utf-8")
    cnn_readme = destination / "resnet18" / "README.md"
    cnn_readme.write_text(cnn_readme.read_text(encoding="utf-8").replace(
        "../role6_submission/README.md", "ROLE6_FINE_TUNED_EXPERIMENTS.md"), encoding="utf-8")
    archive = repo / "CSCI435_Role6_Submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for file in sorted(destination.rglob("*")):
            if file.is_file():
                output.write(file, file.relative_to(destination).as_posix())
    with zipfile.ZipFile(repo / "CSCI435_Role6_Selected_Checkpoints.zip", "w", zipfile.ZIP_DEFLATED) as output:
        for seed in (11, 22, 33, 44, 55):
            run = repo / "Dhan-Shomadhan" / "experiments" / "finetuned_resnet18" / f"seed_{seed}"
            config = json.loads((run / "run_config.json").read_text())
            relative = Path(config["selected_checkpoint"].replace("\\", "/"))
            base = Path("Dhan-Shomadhan/experiments/finetuned_resnet18") / f"seed_{seed}"
            output.write(run / relative, (base / relative).as_posix())
            output.write(run / "run_config.json", (base / "run_config.json").as_posix())
    return destination


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(export_role5_layout(root, root / "role6_submission"))
