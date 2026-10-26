"""Group-supplied HSV/LBP SVM baseline, adapted to the shared dataset manifests.

Run from the repository root. See SVM_BASELINE.md for setup and output files.
"""
import argparse
import csv
import hashlib
import json
import pickle
import platform
from pathlib import Path

import cv2
import numpy as np
import PIL
import scipy
import skimage
import sklearn
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_preprocessing.prepare_dataset import (
    BACKGROUNDS, CLASSES, PARTITIONS, SEEDS, RiceDataset,
    sha_file, validate_split, write_csv, write_json,
)

NUM_CLASSES = 5
IMG_SIZE = 256
HSV_BINS = [16, 4, 4]
LBP_SETTINGS = [(8, 1), (16, 2)]
C_VALUES = [0.1, 1, 10, 100]
GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1]
FEATURE_COUNT = int(np.prod(HSV_BINS)) + sum(points + 2 for points, _ in LBP_SETTINGS)


def hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS,
                        [0, 180, 0, 256, 0, 256]).flatten()
    return hist / hist.sum()


def lbp_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histograms = []
    for points, radius in LBP_SETTINGS:
        lbp = local_binary_pattern(gray, points, radius, method="uniform")
        bins = points + 2
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins))
        histograms.append(hist / hist.sum())
    return np.concatenate(histograms)


def extract_features(img):
    """Preserve the original baseline's resize, blur, HSV and LBP features."""
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected a readable three-channel BGR image")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.concatenate([hsv_histogram(img), lbp_histogram(img)])


def feature_settings():
    return dict(image_size=IMG_SIZE, gaussian_blur=[3, 3], hsv_bins=HSV_BINS,
                lbp_settings=LBP_SETTINGS, lbp_method="uniform", features=FEATURE_COUNT,
                image_loading="shared EXIF-aware RGB loader, converted to BGR for OpenCV",
                code_sha256=sha_file(Path(__file__)),
                versions=dict(numpy=np.__version__, opencv=cv2.__version__,
                              skimage=skimage.__version__, pillow=PIL.__version__))


def feature_signature():
    return hashlib.sha256(json.dumps(feature_settings(), sort_keys=True).encode()).hexdigest()


def load_saved_features(path, signature):
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        if str(data["signature"].item()) != signature:
            raise ValueError("Feature cache settings differ; use a fresh cache")
        keys, features = data["keys"], data["features"]
        if features.shape != (len(keys), FEATURE_COUNT) or not np.isfinite(features).all():
            raise ValueError("Feature cache is incomplete or has invalid values")
        return {str(key): feat for key, feat in zip(keys, features)}


def save_features(path, saved, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(saved)
    np.savez(path, signature=np.array(signature), keys=np.array(keys),
             features=np.array([saved[key] for key in keys]))


def get_features(dataset, saved):
    features = []
    extracted = 0
    for i, row in enumerate(dataset.rows):
        # Check even cache hits: changed image bytes must not reuse stale features.
        if sha_file(dataset.root / row["path"]) != row["sha256"]:
            raise ValueError(f"Source image changed since audit: {row['path']}")
        key = row["sha256"]
        if key not in saved:
            image = dataset[i]["image"]
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            saved[key] = extract_features(bgr)
            extracted += 1
            if extracted % 100 == 0:
                print(f"  extracted {extracted} images", flush=True)
        features.append(saved[key])
    return np.asarray(features)


def load_shared_splits(root, audit, seed):
    version = (audit / (audit / "CURRENT_SPLIT_VERSION.txt").read_text().strip()).resolve()
    if not version.is_relative_to(audit.resolve()):
        raise ValueError("Split version must be inside the shared audit directory")
    provenance = json.loads((version / "provenance.json").read_text())
    datasets, all_rows, assignment = {}, [], {}
    for partition in PARTITIONS:
        relative = f"seed_{seed}/{partition}.csv"
        manifest = version / relative
        if sha_file(manifest) != provenance["manifest_sha256"][relative]:
            raise ValueError(f"Shared manifest checksum changed: {relative}")
        dataset = RiceDataset(root, manifest)
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


def train_svm(X_train, y_train, X_val, y_val, seed):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train)
    scaled_val = scaler.transform(X_val)
    best_model, best_params, best_f1 = None, None, -1
    history = []
    for C in C_VALUES:
        for gamma in GAMMA_VALUES:
            model = SVC(kernel="rbf", C=C, gamma=gamma,
                        class_weight="balanced", random_state=seed)
            model.fit(scaled_train, y_train)
            score = float(f1_score(y_val, model.predict(scaled_val), average="macro",
                                   labels=list(range(NUM_CLASSES)), zero_division=0))
            history.append(dict(C=C, gamma=gamma, validation_macro_f1=score))
            if score > best_f1:  # Preserve first grid candidate in an exact tie.
                best_model, best_params, best_f1 = model, dict(C=C, gamma=gamma), score
    return best_model, scaler, best_params, best_f1, history


def get_scores(y_true, y_pred, backgrounds):
    y_true, y_pred, backgrounds = map(np.asarray, (y_true, y_pred, backgrounds))
    scores = {}
    for name in ("field", "white", "all"):
        mask = np.ones(len(y_true), dtype=bool) if name == "all" else backgrounds == name
        if not mask.any():
            raise ValueError(f"No test images for scenario {name}")
        scores[name] = dict(images=int(mask.sum()),
            accuracy=float(accuracy_score(y_true[mask], y_pred[mask])),
            macro_f1=float(f1_score(y_true[mask], y_pred[mask], average="macro",
                                   labels=list(range(NUM_CLASSES)), zero_division=0)),
            confusion_matrix=confusion_matrix(y_true[mask], y_pred[mask],
                                             labels=list(range(NUM_CLASSES))).tolist())
    return scores


def summary_rows(results):
    rows = []
    for scenario in ("field", "white", "all"):
        acc = [r[scenario]["accuracy"] for r in results]
        f1 = [r[scenario]["macro_f1"] for r in results]
        rows.append(dict(scenario=scenario, runs=len(results), accuracy_mean=float(np.mean(acc)),
                         accuracy_std=float(np.std(acc, ddof=1)) if len(acc) > 1 else "",
                         macro_f1_mean=float(np.mean(f1)),
                         macro_f1_std=float(np.std(f1, ddof=1)) if len(f1) > 1 else ""))
    return rows


def run(root, audit, output, seeds, validation_only=False):
    root, audit, output = root.resolve(), audit.resolve(), output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("Save generated run files under the main dataset directory")
    if not seeds or len(seeds) != len(set(seeds)) or any(s not in SEEDS for s in seeds):
        raise ValueError(f"Choose distinct seeds from {SEEDS}")
    destinations = [output / f"seed_{seed}" for seed in seeds]
    if any(p.exists() for p in destinations):
        raise ValueError("A selected run directory already exists; choose another --output directory")
    signature = feature_signature()
    cache = root / "svm_cache" / f"features_{signature[:16]}.npz"
    saved = load_saved_features(cache, signature)
    results = []
    for seed, directory in zip(seeds, destinations):
        print(f"\n=== Seed {seed}", flush=True)
        datasets, version = load_shared_splits(root, audit, seed)
        X_train = get_features(datasets["train"], saved)
        X_val = get_features(datasets["validation"], saved)
        y_train = [int(r["disease_id"]) for r in datasets["train"].rows]
        y_val = [int(r["disease_id"]) for r in datasets["validation"].rows]
        save_features(cache, saved, signature)
        model, scaler, params, val_f1, history = train_svm(X_train, y_train, X_val, y_val, seed)
        directory.mkdir(parents=True)
        write_csv(directory / "validation_history.csv", history)
        with (directory / "selected_model.pkl").open("wb") as f:
            pickle.dump(dict(model=model, scaler=scaler, class_mapping=CLASSES), f)
        config = dict(method="baseline", seed=seed, manifest_version=version.relative_to(audit).as_posix(),
            code_sha256=sha_file(Path(__file__)), feature_settings=feature_settings(),
            feature_signature=signature, selected_parameters=params,
            parameter_grid=dict(C=C_VALUES, gamma=GAMMA_VALUES), kernel="rbf", class_weight="balanced",
            selection_metric="validation macro-F1", best_validation_macro_f1=val_f1,
            scaler_fit="training partition only; selected model is not refit on train+validation",
            validation_only=validation_only, class_mapping=CLASSES, background_mapping=BACKGROUNDS,
            python=platform.python_version(), sklearn=sklearn.__version__, scipy=scipy.__version__,
            model_file="selected_model.pkl",
            manifest_sha256={part: sha_file(version / f"seed_{seed}" / f"{part}.csv") for part in PARTITIONS})
        write_json(directory / "run_config.json", config)
        print(f"Best parameters: {params}; validation macro-F1: {val_f1:.4f}", flush=True)
        if validation_only:
            continue
        X_test = get_features(datasets["test"], saved)
        save_features(cache, saved, signature)
        y_pred = model.predict(scaler.transform(X_test))
        predictions = [dict(image_id=r["image_id"], seed=seed, method="baseline",
                            disease_id=int(r["disease_id"]), predicted_disease_id=int(pred),
                            background_id=int(r["background_id"]))
                       for r, pred in zip(datasets["test"].rows, y_pred)]
        write_csv(directory / "test_predictions.csv", predictions)
        scores = get_scores([p["disease_id"] for p in predictions], y_pred,
                            [r["background"] for r in datasets["test"].rows])
        write_json(directory / "metrics.json", scores)
        results.append(scores)
        for name, score in scores.items():
            print(f"{name}: accuracy {score['accuracy']:.4f}; macro-F1 {score['macro_f1']:.4f}")
    if results:
        # Only summarise this invocation; do not silently mix different settings.
        filename = "summary_" + "_".join(map(str, seeds)) + ".csv"
        write_csv(output / filename, summary_rows(results))
    print(f"\nSaved run files under {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Dhan-Shomadhan"))
    parser.add_argument("--audit", type=Path, help="Default: repository's Dhan-Shomadhan/preprocessing")
    parser.add_argument("--output", type=Path, help="Default: DATASET/experiments/baseline[_validation]")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--validation-only", action="store_true", help="Select on train/validation without scoring test images")
    args = parser.parse_args()
    audit = args.audit or Path(__file__).resolve().parent / "Dhan-Shomadhan" / "preprocessing"
    name = "baseline_validation" if args.validation_only else "baseline"
    output = args.output or args.dataset / "experiments" / name
    run(args.dataset, audit, output, args.seeds, args.validation_only)


if __name__ == "__main__":
    main()
