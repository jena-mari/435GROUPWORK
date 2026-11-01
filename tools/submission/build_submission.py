"""Build a standalone submission from the repository's actual source and records.

No raw images, previews, feature caches, environments or checkpoints are bundled.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path


def encode(path):
    return base64.b64encode(path.read_bytes()).decode('ascii')


def build(repo, template, output, thumbnail):
    modules = {}
    for folder in ('data_preprocessing', 'hsv_lbp_svm', 'resnet18', 'evaluation'):
        for path in sorted((repo / folder).rglob('*.py')):
            if '__pycache__' not in path.parts and 'tests' not in path.parts and not path.name.startswith('test_'):
                modules[path.relative_to(repo).as_posix()] = encode(path)
    modules['data_preprocessing/__init__.py'] = ''
    modules['evaluation/thumbnail_experiment.py'] = encode(thumbnail)
    audit_root = repo / 'Dhan-Shomadhan/preprocessing'
    version = (audit_root / 'CURRENT_SPLIT_VERSION.txt').read_text().strip()
    audit = {}
    for name in ('CURRENT_SPLIT_VERSION.txt', 'audit_summary.json', 'inventory.csv',
                 'metadata_check.json', 'counts_raw.csv', 'pair_review.csv', 'image_decisions.csv',
                 'exact_duplicates.csv', 'strongest_pairs.csv'):
        path = audit_root / name
        audit[name] = encode(path)
    for path in sorted((audit_root / version).rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.csv', '.txt'):
            audit[path.relative_to(audit_root).as_posix()] = encode(path)
    recorded = {}
    for method in ('baseline', 'frozen_resnet18', 'finetuned_resnet18'):
        for path in sorted((repo / 'results' / method).glob('seed_*/*')):
            if path.is_file() and path.suffix in ('.csv', '.json'):
                recorded[path.relative_to(repo / 'results').as_posix()] = encode(path)
    for name in ('test_predictions.csv', 'verification.json', 'metrics_by_run.csv', 'summary.csv'):
        recorded['thumbnail_reference/' + name] = encode(repo / 'evaluation/reference/role4_thumbnail_svm' / name)
    requirements = {'svm.txt': encode(repo / 'hsv_lbp_svm/requirements.txt')}
    cnn = b'\n'.join((repo / name).read_bytes() for name in
                    ('data_preprocessing/requirements.txt', 'resnet18/requirements.txt', 'evaluation/requirements.txt'))
    requirements['cnn.txt'] = base64.b64encode(cnn).decode('ascii')
    entries = ['data_preprocessing.verify_dataset', 'data_preprocessing.prepare_dataset',
               'resnet18.check_cnn', 'resnet18.run_role6', 'hsv_lbp_svm.svm_baseline',
               'evaluation.role4_evaluation', 'evaluation.thumbnail_experiment']
    provenance = {name: hashlib.sha256(base64.b64decode(data)).hexdigest()
                  for name, data in modules.items()}
    modules['source_manifest.json'] = base64.b64encode(json.dumps(provenance, indent=2).encode()).decode()
    data = json.dumps(dict(modules=modules, audit=audit, recorded=recorded,
                           requirements=requirements, entry_modules=entries), sort_keys=True).encode()
    payload = base64.b64encode(gzip.compress(data, mtime=0)).decode('ascii')
    source = template.read_text().replace('__BUNDLED_PAYLOAD__', repr(payload)).replace(
        '__BUNDLED_SHA256__', repr(hashlib.sha256(data).hexdigest()))
    compile(source, str(output), 'exec')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    print(f'Built {output}: {len(modules)} source files, {len(audit)} audit files, '
          f'{len(recorded)} recorded outputs; {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=here / 'entrypoint.py.in')
    parser.add_argument('--thumbnail', type=Path)
    parser.add_argument('--output', type=Path, default=here / 'group_name.py')
    args = parser.parse_args()
    thumbnail = args.thumbnail or args.repo / 'evaluation/thumbnail_experiment.py'
    if not thumbnail.is_file() and args.thumbnail is None:
        thumbnail = here / 'thumbnail_experiment.py'
    build(args.repo.resolve(), args.template, args.output, thumbnail)
