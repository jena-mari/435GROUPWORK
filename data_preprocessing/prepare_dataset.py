"""Audit Dhan-Shomadhan and create shared, group-disjoint repeated holdouts.

Python 3.12; NumPy and Pillow. No images are edited or deleted.
Run `python prepare_dataset.py --help` for the two-stage workflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps

CLASSES = {"brown_spot": 0, "leaf_scald": 1, "rice_blast": 2,
           "rice_tungro": 3, "sheath_blight": 4}
BACKGROUNDS = {"field": 0, "white": 1}
DISEASE_ALIASES = {
    "brown spot": "brown_spot", "browon spot": "brown_spot",
    "leaf scaled": "leaf_scald", "leaf scald": "leaf_scald",
    "rice blast": "rice_blast", "rice turgro": "rice_tungro",
    "rice tungro": "rice_tungro", "shath blight": "sheath_blight",
    "sheath blight": "sheath_blight"}
BACKGROUND_ALIASES = {"field background": "field", "white background": "white"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SEEDS = (11, 22, 33, 44, 55)
RATIOS = np.array([.70, .15, .15])
PARTITIONS = ("train", "validation", "test")
RESAMPLE = Image.Resampling.LANCZOS


def norm(text):
    return " ".join(text.casefold().split())


def sha_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rgb_image(path):
    # verify() checks the stream; reopening and load() force complete decoding.
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


def pack_hash(bits):
    return f"{int(''.join('1' if v else '0' for v in bits.ravel()), 2):016x}"


def perceptual_hashes(im):
    # dHash: 64 horizontal comparisons on a 9x8 grayscale thumbnail.
    a = np.asarray(im.convert("L").resize((9, 8), RESAMPLE))
    dhash = pack_hash(a[:, 1:] > a[:, :-1])
    # pHash: orthonormal DCT-II on 32x32 grayscale; top-left 8x8, AC median.
    a = np.asarray(im.convert("L").resize((32, 32), RESAMPLE), dtype=float)
    k = np.arange(32)[:, None]
    c = np.cos(np.pi * (np.arange(32)[None, :] + .5) * k / 32) / 4
    c[0] /= np.sqrt(2)
    low = (c @ a @ c.T)[:8, :8]
    phash = pack_hash(low > np.median(low.ravel()[1:]))
    return dhash, phash


def label_path(relative):
    # Never infer labels from filename prefixes; preserve source folder text.
    parts = relative.parts
    if len(parts) != 3:
        return "", ""
    return DISEASE_ALIASES.get(norm(parts[1]), ""), BACKGROUND_ALIASES.get(norm(parts[0]), "")


def inspect_one(root, path, output):
    relative = path.relative_to(root)
    disease, background = label_path(relative)
    row = dict(image_id=digest(relative.as_posix())[:20], path=relative.as_posix(),
               source_disease=relative.parent.name, source_background=relative.parts[0],
               disease=disease, background=background, sha256=sha_file(path),
               readable=0, width="", height="", source_mode="", dhash="", phash="",
               mean_luminance="", luminance_sd="", laplacian_variance_256="", error="")
    try:
        with Image.open(path) as original:
            row["source_mode"] = original.mode
        im = rgb_image(path)
        row.update(readable=1, width=im.width, height=im.height)
        row["dhash"], row["phash"] = perceptual_hashes(im)
        gray = np.asarray(im.convert("L").resize((256, 256), RESAMPLE), dtype=float)
        lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
               + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
        row.update(mean_luminance=round(float(gray.mean()), 3),
                   luminance_sd=round(float(gray.std()), 3),
                   laplacian_variance_256=round(float(lap.var()), 3))
        im.thumbnail((240, 300), RESAMPLE)
        im.save(output / "thumbnails" / f"{row['image_id']}.jpg", quality=85)
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def counts(rows):
    counter = Counter((r["disease"], r["background"]) for r in rows)
    return [dict(disease=d, background=b, count=counter[d, b])
            for d in CLASSES for b in BACKGROUNDS]


def sample_figure(rows, output, name):
    """One deterministic sample per disease/background, uncropped and unmodified."""
    canvas = Image.new("RGB", (1320, 810), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    draw.text((20, 10), "Figure 1. Dhan-Shomadhan examples (original aspect ratio)", fill="black", font=font)
    for i, disease in enumerate(CLASSES):
        for j, background in enumerate(BACKGROUNDS):
            x, y = 95 + i * 242, 80 + j * 355
            if j == 0:
                draw.text((x, 45), disease.replace('_', ' ').title(), fill="black", font=font)
            if i == 0:
                draw.text((10, y + 140), background.upper(), fill="black", font=small)
            pool = sorted((r for r in rows if r["disease"] == disease
                           and r["background"] == background and str(r["readable"]) == "1"),
                          key=lambda r: r["path"])
            if pool:
                r = pool[0]
                with Image.open(output / "thumbnails" / f"{r['image_id']}.jpg") as im:
                    canvas.paste(im, (x + (230 - im.width) // 2, y))
                draw.text((x + 35, y + 307), Path(r["path"]).name, fill="black", font=small)
    canvas.save(output / name)


def metadata_check(root, inventory):
    path = root / "Dhan-Shomadhan_picture_Information.csv"
    if not path.exists():
        return {"available": False, "issues": []}
    metadata = read_csv(path)
    by_name = defaultdict(list)
    for r in inventory:
        by_name[Path(r["path"]).name].append(r)
    issues, seen = [], set()
    for row in metadata:
        name, text = row.get("pictureName", ""), row.get("Diseases", "")
        disease_text, _, background_text = text.partition("(")
        disease = DISEASE_ALIASES.get(norm(disease_text), "")
        bg = BACKGROUND_ALIASES.get(norm(background_text.rstrip(")")).replace("feild", "field"), "")
        if name in seen or len(by_name[name]) != 1:
            issues.append(dict(filename=name, issue="duplicate metadata name or ambiguous/missing image"))
        elif (disease, bg) != (by_name[name][0]["disease"], by_name[name][0]["background"]):
            issues.append(dict(filename=name, issue="metadata/folder label disagreement"))
        seen.add(name)
    for name in by_name.keys() - seen:
        issues.append(dict(filename=name, issue="image missing from supplied metadata"))
    return {"available": True, "rows": len(metadata), "sha256": sha_file(path), "issues": issues}


def source_images(root, output):
    """Keep existing audit previews and experiment figures out of the source set."""
    generated = {output, root / "preprocessing", root / "experiments"}
    generated.update(p.parent for p in root.glob("*/audit_summary.json"))
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and p.suffix.lower() in EXTENSIONS
                  and not any(p.is_relative_to(directory) for directory in generated)
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


def audit(root, output, workers=4):
    if output.exists():
        raise ValueError("Audit output already exists. Use a new output directory to preserve reviews.")
    (output / "thumbnails").mkdir(parents=True)
    files = source_images(root, output)
    if not files:
        raise ValueError("No images found")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda p: inspect_one(root, p, output), files))
    write_csv(output / "inventory.csv", rows)
    write_csv(output / "counts_raw.csv", counts(rows))
    exact = defaultdict(list)
    for r in rows:
        exact[r["sha256"]].append(r)
    duplicate_rows = [dict(sha256=sha, image_id=r["image_id"], path=r["path"],
                           disease=r["disease"], background=r["background"],
                           label_conflict=int(len({(x['disease'], x['background']) for x in members}) > 1))
                      for sha, members in exact.items() if len(members) > 1 for r in members]
    write_csv(output / "exact_duplicates.csv", duplicate_rows,
              ["sha256", "image_id", "path", "disease", "background", "label_conflict"])
    readable = [r for r in rows if r["readable"]]
    pairs = []
    for i, a in enumerate(readable):
        for b in readable[i + 1:]:
            if a["sha256"] == b["sha256"]:
                continue
            dh = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            ph = (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count()
            if dh <= 6 or ph <= 8:
                pairs.append(dict(left_id=a["image_id"], right_id=b["image_id"],
                                  left_path=a["path"], right_path=b["path"],
                                  dhash_distance=dh, phash_distance=ph,
                                  decision="pending", reason=""))
    write_csv(output / "pair_review.csv", pairs,
              ["left_id", "right_id", "left_path", "right_path", "dhash_distance",
               "phash_distance", "decision", "reason"])
    write_csv(output / "image_decisions.csv", [],
              ["image_id", "action", "disease", "background", "manual_group", "reason"])
    metadata = metadata_check(root, rows)
    write_json(output / "metadata_check.json", metadata)
    label_issues = [r["path"] for r in rows if not r["disease"] or not r["background"]]
    summary = dict(images=len(rows), readable=sum(r["readable"] for r in rows),
                   unreadable=sum(not r["readable"] for r in rows),
                   unknown_label_paths=label_issues, metadata_issues=len(metadata["issues"]),
                   exact_duplicate_groups=sum(len(x) > 1 for x in exact.values()),
                   exact_extra_copies=sum(max(0, len(x) - 1) for x in exact.values()),
                   near_duplicate_candidates=len(pairs),
                   dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
                   class_mapping=CLASSES, background_mapping=BACKGROUNDS,
                   disease_aliases=DISEASE_ALIASES, background_aliases=BACKGROUND_ALIASES,
                   inventory_sha256=sha_file(output / "inventory.csv"),
                   source_root=str(root), python=platform.python_version(),
                   numpy=np.__version__, pillow=PIL.__version__,
                   candidate_rule="dHash64 Hamming <= 6 OR pHash64 Hamming <= 8; original orientation",
                   candidate_warning="Fixed screening thresholds, not calibrated same-leaf detection. Review all candidates; misses remain possible.")
    write_json(output / "audit_summary.json", summary)
    sample_figure(rows, output, "samples_raw.png")
    make_review_pages(rows, pairs, output)
    print(json.dumps({k: summary[k] for k in ["images", "unreadable", "metadata_issues",
          "exact_duplicate_groups", "near_duplicate_candidates", "dimensions"]}, indent=2))


def make_review_pages(rows, pairs, output):
    # Standalone HTML uses audit thumbnails only; no script or network dependency.
    import html
    cards = []
    for i, p in enumerate(pairs):
        images = "".join(f'<figure><img src="thumbnails/{p[side+"_id"]}.jpg">'
                         f'<figcaption>{html.escape(p[side+"_path"])}</figcaption></figure>'
                         for side in ("left", "right"))
        cards.append(f'<article><h3>Pair {i + 1}: dHash {p["dhash_distance"]}, '
                     f'pHash {p["phash_distance"]}</h3><div>{images}</div></article>')
    (output / "review.html").write_text('<!doctype html><meta charset="utf-8">'
        '<title>Near-duplicate review</title><style>body{font:16px system-ui;max-width:1000px;margin:auto}'
        'article{border-top:1px solid #aaa;padding:16px}article div{display:flex}'
        'figure{width:45%;margin:8px}img{max-height:300px}figcaption{overflow-wrap:anywhere}</style>'
        '<h1>Near-duplicate candidates</h1><p>Edit pair_review.csv after inspection: related, '
        'distinct, or conservative. Add a reason. Conservative keeps uncertain pairs together; '
        'it does not claim that the same leaf was confirmed.</p>' + ''.join(cards))
    for disease in CLASSES:
        pool = [r for r in rows if r["disease"] == disease]
        cards = ''.join(f'<figure><img src="thumbnails/{r["image_id"]}.jpg">'
                        f'<figcaption>{html.escape(r["path"])}</figcaption></figure>' for r in pool)
        (output / f"gallery_{disease}.html").write_text('<!doctype html><meta charset="utf-8">'
            '<style>body{font:14px system-ui}main{display:flex;flex-wrap:wrap}'
            'figure{width:240px;margin:12px}img{max-width:240px}figcaption{overflow-wrap:anywhere}</style>'
            f'<h1>{disease}</h1><p>Inspect for repeated leaves missed by hash screening. '
            'Use image_decisions.csv manual_group for additional related sets.</p><main>' + cards + '</main>')


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, a, b):
        x, y = sorted([self.find(a), self.find(b)])
        self.parent[y] = x


def resolve_inventory(rows, decisions, pairs):
    rows = [dict(r) for r in rows]
    by_id = {r["image_id"]: r for r in rows}
    log, excluded, manual = [], set(), defaultdict(list)
    used = set()
    for change in decisions:
        i = change["image_id"]
        if i not in by_id or i in used:
            raise ValueError(f"Unknown or repeated decision ID {i}")
        used.add(i)
        if not change.get("reason", "").strip():
            raise ValueError("Every image decision requires a reason")
        if change["action"] not in ("include", "exclude"):
            raise ValueError("Image action must be include/exclude")
        if change["action"] == "exclude":
            excluded.add(i)
        for key in ("disease", "background"):
            if change.get(key):
                by_id[i][key] = change[key]
        if change.get("manual_group"):
            manual[change["manual_group"]].append(i)
        log.append(dict(image_id=i, action=change["action"], reason=change["reason"]))
    for r in rows:
        i = r["image_id"]
        if str(r["readable"]) != "1":
            excluded.add(i)
            log.append(dict(image_id=i, action="exclude", reason="unreadable image"))
        if i not in excluded and (r["disease"] not in CLASSES or r["background"] not in BACKGROUNDS):
            raise ValueError(f"Unresolved label: {r['path']}")
    uf = UnionFind(by_id)
    for members in manual.values():
        for i in members[1:]:
            uf.union(members[0], i)
    for p in pairs:
        a, b = p["left_id"], p["right_id"]
        if a not in by_id or b not in by_id:
            raise ValueError("Unknown pair ID")
        if a in excluded or b in excluded:
            continue
        if p["decision"] not in ("related", "distinct", "conservative") or not p["reason"].strip():
            raise ValueError(f"Pending/undocumented pair review: {a}, {b}")
        if p["decision"] in ("related", "conservative"):
            uf.union(a, b)
    exact = defaultdict(list)
    for r in rows:
        if r["image_id"] not in excluded:
            exact[r["sha256"]].append(r)
    for members in exact.values():
        if len({(r["disease"], r["background"]) for r in members}) > 1:
            raise ValueError("Exact-copy label conflict requires an explicit image decision")
        members.sort(key=lambda r: r["path"])
        for r in members[1:]:
            uf.union(members[0]["image_id"], r["image_id"])
            excluded.add(r["image_id"])
            log.append(dict(image_id=r["image_id"], action="exclude",
                            reason=f"exact SHA-256 copy of {members[0]['image_id']}"))
    eligible = [r for r in rows if r["image_id"] not in excluded]
    for r in eligible:
        r.update(group_id="g_" + uf.find(r["image_id"]), disease_id=CLASSES[r["disease"]],
                 background_id=BACKGROUNDS[r["background"]])
    return eligible, log


def grouped_split(rows, seed, trials=200):
    """Seeded multi-start greedy allocation, minimising 10-stratum deviation.

    Groups are indivisible; exact percentages need not be attainable. Require each
    observed disease/background stratum in every partition or fail visibly.
    """
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["group_id"]].append(r)
    ids = sorted(grouped)
    vectors = np.zeros((len(ids), 10), dtype=int)
    for i, gid in enumerate(ids):
        for r in grouped[gid]:
            vectors[i, CLASSES[r["disease"]] * 2 + BACKGROUNDS[r["background"]]] += 1
    total = vectors.sum(axis=0)
    if np.any(total == 0):
        raise ValueError("All ten disease/background strata are required for this experiment")
    if np.any((vectors > 0).sum(axis=0) < 3):
        raise ValueError("Fewer than 3 independent groups in a stratum: requested coverage impossible")
    targets = RATIOS[:, None] * total[None, :]
    # Weight relative errors without letting large classes dominate every decision.
    denominator = np.maximum(targets, 1)
    rng = random.Random(seed)
    best = None
    for _ in range(trials):
        order = list(range(len(ids)))
        rng.shuffle(order)
        order.sort(key=lambda i: -vectors[i].sum())  # stable random ties
        achieved = np.zeros((3, 10), dtype=int)
        allocation = np.empty(len(ids), dtype=int)
        for i in order:
            candidates = list(range(3))
            rng.shuffle(candidates)
            costs = []
            for partition in candidates:
                before = ((achieved[partition] - targets[partition]) ** 2 / denominator[partition]).sum()
                after = ((achieved[partition] + vectors[i] - targets[partition]) ** 2 / denominator[partition]).sum()
                costs.append((float(after - before), partition))
            _, partition = min(costs, key=lambda v: v[0])
            allocation[i] = partition
            achieved[partition] += vectors[i]
        if np.any(achieved == 0):
            continue
        cost = float(((achieved - targets) ** 2 / denominator).sum())
        if best is None or cost < best[0]:
            best = cost, allocation.copy(), achieved.copy()
    if best is None:
        raise ValueError("Allocator could not achieve required coverage; review groups or change design explicitly")
    _, allocation, achieved = best
    assignment = {gid: PARTITIONS[int(allocation[i])] for i, gid in enumerate(ids)}
    return assignment, achieved.tolist()


def validate_split(rows, assignment):
    if set(assignment) != {r["group_id"] for r in rows}:
        raise ValueError("Group assignment does not cover inventory exactly")
    seen, hashes = set(), {}
    for r in rows:
        if r["image_id"] in seen:
            raise ValueError("Repeated image ID")
        seen.add(r["image_id"])
        part = assignment[r["group_id"]]
        if part not in PARTITIONS:
            raise ValueError("Invalid partition")
        if r["sha256"] in hashes and hashes[r["sha256"]] != part:
            raise ValueError("Exact duplicate leakage")
        hashes[r["sha256"]] = part
    for part in PARTITIONS:
        subset = [r for r in rows if assignment[r["group_id"]] == part]
        if {(r["disease"], r["background"]) for r in subset} != {(d, b) for d in CLASSES for b in BACKGROUNDS}:
            raise ValueError("Missing disease/background stratum in a partition")


def build(root, output, gallery_reviewed=False, allow_metadata_issues=False,
          conservative_pending=False):
    if not gallery_reviewed and not conservative_pending:
        raise ValueError("Inspect disease galleries, record additional manual groups, then use --gallery-reviewed")
    summary = json.loads((output / "audit_summary.json").read_text())
    if sha_file(output / "inventory.csv") != summary["inventory_sha256"]:
        raise ValueError("Inventory changed: regenerate audit; record corrections in image_decisions.csv")
    rows = read_csv(output / "inventory.csv")
    source_paths = {p.relative_to(root).as_posix() for p in source_images(root, output)}
    if source_paths != {r["path"] for r in rows}:
        raise ValueError("Source file set changed since audit")
    for r in rows:
        if sha_file(root / r["path"]) != r["sha256"]:
            raise ValueError(f"Image changed since audit: {r['path']}")
    metadata = metadata_check(root, rows)
    saved_metadata = json.loads((output / "metadata_check.json").read_text())
    if metadata != saved_metadata:
        raise ValueError("Source metadata changed since audit")
    if metadata["issues"] and not allow_metadata_issues:
        raise ValueError("Resolve metadata issues or document them and use --allow-metadata-issues")
    pairs, decisions = read_csv(output / "pair_review.csv"), read_csv(output / "image_decisions.csv")
    # Explicit alternative: exploratory splits can conservatively join unresolved
    # candidate pairs without claiming that a visual review has been completed.
    pending_grouped = 0
    if conservative_pending:
        for p in pairs:
            if p['decision'] == 'pending':
                p['decision'] = 'conservative'
                p['reason'] = 'Unresolved automated candidate; kept together conservatively, not visually confirmed'
                pending_grouped += 1
    # Preserve every original candidate even when the review file is edited.
    readable = [r for r in rows if r['readable'] == '1']
    required = {tuple(sorted((a['image_id'], b['image_id'])))
                for i, a in enumerate(readable) for b in readable[i + 1:]
                if a['sha256'] != b['sha256'] and
                ((int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count() <= 6 or
                 (int(a['phash'], 16) ^ int(b['phash'], 16)).bit_count() <= 8)}
    supplied = [tuple(sorted((p['left_id'], p['right_id']))) for p in pairs]
    if not required.issubset(supplied) or len(set(supplied)) != len(supplied):
        raise ValueError("Candidate review rows removed or duplicated")
    eligible, log = resolve_inventory(rows, decisions, pairs)
    # Build all five in memory before publishing any manifest.
    runs = []
    for seed in SEEDS:
        assignment, achieved = grouped_split(eligible, seed)
        validate_split(eligible, assignment)
        runs.append((seed, assignment, achieved))
    fingerprint = digest(dict(inventory=summary["inventory_sha256"],
                              pairs=sha_file(output / "pair_review.csv"),
                              decisions=sha_file(output / "image_decisions.csv"),
                              code=sha_file(Path(__file__)), classes=CLASSES,
                              backgrounds=BACKGROUNDS, ratios=RATIOS.tolist(),
                              gallery_reviewed=gallery_reviewed,
                              conservative_pending=conservative_pending))
    destination = output / ("conservative_splits" if conservative_pending else "splits") / fingerprint[:16]
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "eligible_inventory.csv", eligible)
    write_csv(destination / "exclusions_and_corrections.csv", log, ["image_id", "action", "reason"])
    write_csv(destination / "counts_eligible.csv", counts(eligible))
    write_csv(destination / "effective_pair_decisions.csv", pairs,
              ["left_id", "right_id", "left_path", "right_path", "dhash_distance",
               "phash_distance", "decision", "reason"])
    write_json(destination / "label_mapping.json", dict(disease=CLASSES, background=BACKGROUNDS))
    split_summary, manifest_hashes = [], {}
    for seed, assignment, achieved in runs:
        directory = destination / f"seed_{seed}"
        directory.mkdir(exist_ok=True)
        for part in PARTITIONS:
            subset = [dict(r, partition=part, seed=seed) for r in eligible if assignment[r["group_id"]] == part]
            path = directory / f"{part}.csv"
            write_csv(path, subset)
            manifest_hashes[path.relative_to(destination).as_posix()] = sha_file(path)
            split_summary.append(dict(seed=seed, partition=part, images=len(subset),
                                      groups=len({r["group_id"] for r in subset}),
                                      field=sum(r["background"] == "field" for r in subset),
                                      white=sum(r["background"] == "white" for r in subset)))
            write_csv(directory / f"{part}_counts.csv", counts(subset))
    write_csv(destination / "split_summary.csv", split_summary)
    write_json(destination / "provenance.json", dict(fingerprint=fingerprint, seeds=SEEDS,
        target_ratios=RATIOS.tolist(), algorithm="200 seeded greedy starts; squared stratum errors / target",
        inventory_sha256=summary["inventory_sha256"], manifest_sha256=manifest_hashes,
        source_images=len(rows), eligible_images=len(eligible),
        pair_decisions=dict(Counter(p['decision'] for p in pairs)),
        status="conservative_grouping_with_unresolved_candidates" if conservative_pending else "reviewed",
        unresolved_pairs_grouped=pending_grouped,
        gallery_reviewed=gallery_reviewed, metadata_issues_acknowledged=allow_metadata_issues,
        limitation="No plant/session IDs supplied. Visual review cannot establish complete biological independence.",
        python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__))
    sample_figure(eligible, output, "samples_eligible.png")
    print(f"Saved five validated {'conservatively grouped ' if conservative_pending else ''}repeated holdout splits: {destination}")
    return destination


class RiceDataset:
    """Map-style dataset compatible with torch.utils.data.DataLoader.

    Without a transform, image is an RGB PIL image. Supply training transforms
    only to a train manifest; supply deterministic transforms for evaluation.
    Background is returned as metadata, never concatenated to image features.
    """
    def __init__(self, dataset_root, manifest, transform=None, verify_hashes=False):
        self.root = Path(dataset_root).resolve()
        self.rows = read_csv(manifest)
        self.transform = transform
        if not self.rows:
            raise ValueError("Empty manifest")
        ids = set()
        for r in self.rows:
            path = (self.root / r['path']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Invalid manifest path: {r['path']}")
            if int(r['disease_id']) != CLASSES[r['disease']] or int(r['background_id']) != BACKGROUNDS[r['background']]:
                raise ValueError("Manifest label mapping mismatch")
            if r['image_id'] in ids:
                raise ValueError("Repeated image in manifest")
            ids.add(r['image_id'])
            if verify_hashes and sha_file(path) != r['sha256']:
                raise ValueError(f"Image changed: {r['path']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = rgb_image(self.root / row["path"])
        if self.transform is not None:
            image = self.transform(image)
        return dict(image=image, disease_id=int(row["disease_id"]),
                    background_id=int(row["background_id"]), image_id=row["image_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "build"])
    parser.add_argument("--root", type=Path, default=Path("Dhan-Shomadhan"))
    parser.add_argument("--output", type=Path, help="Default ROOT/preprocessing; temporary external output is allowed")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gallery-reviewed", action="store_true")
    parser.add_argument("--allow-metadata-issues", action="store_true")
    parser.add_argument("--conservative-pending", action="store_true",
                        help="Save conservative splits grouping unresolved candidates, without claiming full visual review")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "preprocessing").resolve()
    if args.stage == "audit":
        audit(root, output, args.workers)
    else:
        build(root, output, args.gallery_reviewed, args.allow_metadata_issues, args.conservative_pending)


if __name__ == "__main__":
    main()
