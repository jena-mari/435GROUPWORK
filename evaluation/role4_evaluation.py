"""Role 4 evaluation, adapted to the group's versioned manifests and outputs.

All five seeds and exact test membership are required. No models are trained.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from importlib.metadata import version as package_version
import os
import tempfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, precision_recall_fscore_support,
)

from data_preprocessing.prepare_dataset import (
    BACKGROUNDS, CLASSES, SEEDS, sha_file, write_csv, write_json,
)

AUDIT = Path(__file__).resolve().parents[1] / 'Dhan-Shomadhan/preprocessing'
LABELS = list(range(5))
CLASS_NAMES = ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
SCENARIOS = ('white', 'field', 'all')
METRICS = ('accuracy', 'precision', 'recall', 'macro_f1')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def shared_tests(audit):
    audit = Path(audit).resolve()
    version = (audit / (audit / 'CURRENT_SPLIT_VERSION.txt').read_text().strip()).resolve()
    if not version.is_relative_to(audit):
        raise ValueError('Split version is outside the audit directory')
    provenance = json.loads((version / 'provenance.json').read_text())
    manifests, hashes = {}, {}
    for seed in SEEDS:
        hashes[seed] = {}
        for partition in ('train', 'validation', 'test'):
            relative = f'seed_{seed}/{partition}.csv'
            path = version / relative
            actual = sha_file(path)
            if actual != provenance['manifest_sha256'][relative]:
                raise ValueError(f'Manifest checksum mismatch: {relative}')
            hashes[seed][partition] = actual
            if partition == 'test':
                rows = read_csv(path)
                if not rows or any(int(r['seed']) != seed or r['partition'] != 'test' for r in rows):
                    raise ValueError('Invalid test manifest')
                manifests[seed] = {r['image_id']: r for r in rows}
                if len(manifests[seed]) != len(rows):
                    raise ValueError('Repeated test-manifest image ID')
    return version.relative_to(audit).as_posix(), manifests, hashes


def normalise_predictions(raw, expected, seed, method, *, reference=False):
    """Normalise only the historical thumbnail schema or the current contract."""
    if not raw:
        raise ValueError(f'Empty predictions for {method}, seed {seed}')
    result, seen = [], set()
    for row in raw:
        image_id = row['image_id']
        if image_id in seen or image_id not in expected:
            raise ValueError(f'Unexpected or duplicate image ID: {image_id}')
        seen.add(image_id)
        source = expected[image_id]
        if int(row['seed']) != seed:
            raise ValueError('Prediction seed disagrees with its run')
        if reference:
            truth, predicted = int(row['true_label']), int(row['pred_label'])
            background = BACKGROUNDS[row['background'].strip().lower()]
        else:
            if row['method'] != method:
                raise ValueError('Prediction method disagrees with its source directory')
            truth, predicted = int(row['disease_id']), int(row['predicted_disease_id'])
            background = int(row['background_id'])
        if truth not in LABELS or predicted not in LABELS or background not in BACKGROUNDS.values():
            raise ValueError('Invalid disease or background identifier')
        if (truth, background) != (int(source['disease_id']), int(source['background_id'])):
            raise ValueError(f'Prediction labels/background disagree with manifest: {image_id}')
        result.append(dict(image_id=image_id, seed=seed, method=method, disease_id=truth,
            predicted_disease_id=predicted, background_id=background, path=source['path']))
    if seen != set(expected):
        raise ValueError(f'Missing predictions for seed {seed}: {len(set(expected) - seen)} images')
    return sorted(result, key=lambda row: row['image_id'])


def check_run_config(path, seed, method, version, hashes):
    config = json.loads(Path(path).read_text())
    context = config.get('context', config)
    if config.get('method') != method or context.get('seed') != seed:
        raise ValueError(f'Run method/seed mismatch: {path}')
    if context.get('manifest_version') != version or context.get('manifest_sha256') != hashes:
        raise ValueError(f'Run manifest version/checksums mismatch: {path}')
    if context.get('class_mapping') != CLASSES or context.get('background_mapping') != BACKGROUNDS:
        raise ValueError(f'Run label mapping mismatch: {path}')
    if method != 'baseline':
        if context.get('run_scope') != 'experiment' or config.get('smoke') or config.get('check_only'):
            raise ValueError('Diagnostic CNN runs cannot enter the final comparison')
        if config.get('weights_source') != 'IMAGENET1K_V1':
            raise ValueError('CNN run does not record ImageNet initialisation')
    return config


def metrics(rows):
    truth = [r['disease_id'] for r in rows]
    predicted = [r['predicted_disease_id'] for r in rows]
    return dict(n=len(rows), accuracy=float(accuracy_score(truth, predicted)),
        precision=float(precision_score(truth, predicted, labels=LABELS, average='macro', zero_division=0)),
        recall=float(recall_score(truth, predicted, labels=LABELS, average='macro', zero_division=0)),
        macro_f1=float(f1_score(truth, predicted, labels=LABELS, average='macro', zero_division=0)),
        matrix=confusion_matrix(truth, predicted, labels=LABELS).tolist())


def subset(rows, scenario):
    selected = rows if scenario == 'all' else [r for r in rows if r['background_id'] == BACKGROUNDS[scenario]]
    if not selected:
        raise ValueError(f'No {scenario} predictions')
    return selected


def summarise(individual):
    summary = []
    for method in sorted({r['method'] for r in individual}):
        for scenario in SCENARIOS:
            runs = [r for r in individual if r['method'] == method and r['scenario'] == scenario]
            if len(runs) != 5 or {r['seed'] for r in runs} != set(SEEDS):
                raise ValueError('Five distinct shared seeds are required for every summary')
            row = dict(method=method, scenario=scenario, runs=5)
            for metric in METRICS:
                values = np.array([r[metric] for r in runs])
                row[metric + '_mean'] = float(values.mean())
                row[metric + '_std'] = float(values.std(ddof=1))
            summary.append(row)
    return summary


def plot_matrix(matrix, title, path):
    # Import lazily so metric checks can run without a display or plot dependencies.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / '435-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=35, cmap='Blues', colorbar=False)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_sources(sources, audit, output, *, thumbnail_reference=None, plots=True):
    output = Path(output)
    if output.exists():
        raise ValueError('Evaluation output exists; choose a new --out directory')
    if bool(sources) == bool(thumbnail_reference):
        raise ValueError('Provide final model sources OR the preliminary thumbnail reference')
    version, manifests, hashes = shared_tests(audit)
    collected, records, policies = {}, [], []
    if thumbnail_reference:
        source = Path(thumbnail_reference) / 'test_predictions.csv'
        raw = read_csv(source)
        if {int(r['seed']) for r in raw} != set(SEEDS):
            raise ValueError('Thumbnail reference must contain exactly the five shared seeds')
        method = 'thumbnail_svm_preliminary'
        for seed in SEEDS:
            collected[(method, seed)] = normalise_predictions(
                [r for r in raw if int(r['seed']) == seed], manifests[seed], seed, method, reference=True)
        records.append(dict(method=method, predictions_sha256=sha_file(source), source=str(source.resolve())))
    else:
        for method, root in sources.items():
            for seed in SEEDS:
                directory = Path(root) / f'seed_{seed}'
                prediction_path = directory / 'test_predictions.csv'
                config_path = directory / 'run_config.json'
                config = check_run_config(config_path, seed, method, version, hashes[seed])
                if method != 'baseline':
                    policies.append(config['context']['policy'])
                collected[(method, seed)] = normalise_predictions(
                    read_csv(prediction_path), manifests[seed], seed, method)
                records.append(dict(method=method, seed=seed, source=str(directory.resolve()),
                    predictions_sha256=sha_file(prediction_path), config_sha256=sha_file(config_path)))
        if policies and any(policy != policies[0] for policy in policies):
            raise ValueError('CNN runs used different preprocessing/augmentation policies')
    # All sources must validate before producing report files.
    individual, classwise, matrices, examples = [], [], {}, []
    for (method, seed), rows in collected.items():
        examples.extend(dict(row, correct=row['disease_id'] == row['predicted_disease_id']) for row in rows)
        for scenario in SCENARIOS:
            selected = subset(rows, scenario)
            score = metrics(selected)
            individual.append(dict(method=method, seed=seed, scenario=scenario, n=score['n'],
                                   **{name: score[name] for name in METRICS}))
            matrices[f'{method}/seed_{seed}/{scenario}'] = score['matrix']
            p, r, f, support = precision_recall_fscore_support(
                [row['disease_id'] for row in selected], [row['predicted_disease_id'] for row in selected],
                labels=LABELS, zero_division=0)
            for label in LABELS:
                classwise.append(dict(method=method, seed=seed, scenario=scenario,
                    disease_id=label, disease=list(CLASSES)[label], precision=float(p[label]),
                    recall=float(r[label]), f1=float(f[label]), support=int(support[label])))
    summary = summarise(individual)
    paired = []
    methods = sorted({method for method, _seed in collected})
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            for scenario in SCENARIOS:
                for metric in METRICS:
                    values = {method: {r['seed']: r[metric] for r in individual
                              if r['method'] == method and r['scenario'] == scenario}
                              for method in (a, b)}
                    differences = [values[b][seed] - values[a][seed] for seed in SEEDS]
                    paired.append(dict(method_a=a, method_b=b, scenario=scenario, metric=metric,
                        difference='method_b minus method_a', mean_difference=float(np.mean(differences)),
                        sample_std_difference=float(np.std(differences, ddof=1)),
                        **{f'seed_{seed}': value for seed, value in zip(SEEDS, differences)}))
    output.mkdir(parents=True)
    write_csv(output / 'per_run_metrics.csv', individual)
    write_csv(output / 'five_run_summary.csv', summary)
    write_csv(output / 'per_class_metrics.csv', classwise)
    write_csv(output / 'prediction_review.csv', examples)
    if paired:
        write_csv(output / 'paired_differences.csv', paired)
    pooled = {}
    for method in methods:
        for scenario in SCENARIOS:
            pooled[f'{method}/{scenario}'] = np.sum([
                matrices[f'{method}/seed_{seed}/{scenario}'] for seed in SEEDS], axis=0).tolist()
    write_json(output / 'confusion_matrices.json', dict(per_run=matrices, pooled=pooled, class_order=CLASS_NAMES))
    write_json(output / 'evaluation_config.json', dict(scope='preliminary_thumbnail_reference' if thumbnail_reference else 'full_resolution_comparison',
        final_group_comparison=not bool(thumbnail_reference) and len(sources) == 3,
        manifest_version=version, manifest_sha256={str(seed): hashes[seed] for seed in SEEDS},
        class_mapping=CLASSES, background_mapping=BACKGROUNDS, seeds=list(SEEDS),
        code_sha256=sha_file(Path(__file__)), python=platform.python_version(),
        numpy=np.__version__, sklearn=sklearn.__version__, plots=plots,
        matplotlib=package_version('matplotlib') if plots else None,
        std_ddof=1, metric_units='fractions', sources=records,
        note='Repeated holdouts overlap. Pooled matrices count test appearances, not unique images. No significance test is performed.'))
    (output / 'README.txt').write_text(
        ('PRELIMINARY THUMBNAIL SVM: not the full-resolution baseline or a CNN result.\n' if thumbnail_reference else
         f'{len(sources)} method(s), five shared seeds each. A full group comparison needs all three methods.\n') +
        'Summary: arithmetic mean and sample SD (ddof=1) of unrounded per-run metrics; multiply by 100 for percentages.\n'
        'Combined metrics use all test predictions directly. Macro metrics always include the five diseases.\n'
        'Confusion matrices use rows=true and columns=predicted. Pooled matrices count repeated test appearances.\n'
        'Use prediction_review.csv to inspect successes and failures in original images. No causes are inferred automatically.\n', encoding='utf-8')
    if plots:
        figures = output / 'figures'
        figures.mkdir()
        for key, matrix in matrices.items():
            plot_matrix(matrix, key.replace('/', ' | '), figures / (key.replace('/', '_') + '_confusion.png'))
        for key, matrix in pooled.items():
            plot_matrix(matrix, key.replace('/', ' | ') + '\nPooled test appearances across five overlapping holdouts',
                        figures / (key.replace('/', '_') + '_pooled_confusion.png'))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--svm', type=Path, help='Canonical experiments/baseline directory')
    parser.add_argument('--frozen', type=Path, help='Canonical experiments/frozen_resnet18 directory')
    parser.add_argument('--finetuned', type=Path, help='Canonical experiments/finetuned_resnet18 directory')
    parser.add_argument('--thumbnail-reference', type=Path, help='Audit the supplied preliminary thumbnail results separately')
    parser.add_argument('--audit', type=Path, default=AUDIT)
    parser.add_argument('--out', type=Path, default=Path('Dhan-Shomadhan/experiments/comparison'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    sources = {name: root for name, root in [('baseline', args.svm), ('frozen_resnet18', args.frozen),
                ('finetuned_resnet18', args.finetuned)] if root is not None}
    summary = evaluate_sources(sources, args.audit, args.out,
                               thumbnail_reference=args.thumbnail_reference, plots=not args.no_plots)
    print(f'Validated all five seeds for each supplied method. Saved {len(summary)} summary rows to {args.out}')


if __name__ == '__main__':
    main()
