"""Verify locally downloaded source images and committed split checksums.

Uses only the Python standard library. Run from the repository root:
    python data_preprocessing/verify_dataset.py
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, audit=None):
    root = root.resolve()
    audit = (audit or root / 'preprocessing').resolve()
    summary = json.loads((audit / 'audit_summary.json').read_text())
    inventory = audit / 'inventory.csv'
    errors = []
    if sha256(inventory) != summary['inventory_sha256']:
        raise ValueError('Inventory checksum mismatch; restore the shared audited inventory.')
    with inventory.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    checked = 0
    for row in rows:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Inventory path is outside the dataset directory.')
        if not path.is_file():
            errors.append(f"Missing: {row['path']}")
        elif sha256(path) != row['sha256']:
            errors.append(f"SHA-256 mismatch: {row['path']}")
        else:
            checked += 1
    metadata = json.loads((audit / 'metadata_check.json').read_text())
    if metadata.get('available'):
        path = root / 'Dhan-Shomadhan_picture_Information.csv'
        if not path.is_file() or sha256(path) != metadata['sha256']:
            errors.append('Source metadata CSV is missing or different.')
    version = (audit / (audit / 'CURRENT_SPLIT_VERSION.txt').read_text().strip()).resolve()
    if not version.is_relative_to(audit):
        raise ValueError('Split version points outside the audit directory.')
    provenance = json.loads((version / 'provenance.json').read_text())
    manifests = 0
    for relative, expected in provenance['manifest_sha256'].items():
        path = (version / relative).resolve()
        if not path.is_relative_to(version):
            raise ValueError('Manifest path is outside the split version directory.')
        if not path.is_file() or sha256(path) != expected:
            errors.append(f'Manifest missing or changed: {relative}')
        else:
            manifests += 1
    if errors:
        print(f'{checked}/{len(rows)} source images match; {manifests} manifest checksums match.')
        for message in errors[:20]:
            print(message)
        if len(errors) > 20:
            print(f'... and {len(errors) - 20} more errors.')
        raise SystemExit('Verification failed. Download version 1 and preserve the original folder paths.')
    print(f'PASS: all {checked} source images, source metadata and {manifests} manifests match the audited checksums.')
    return checked, manifests


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('Dhan-Shomadhan'))
    parser.add_argument('--audit', type=Path, help='Use shared audit files stored separately from the raw dataset')
    args = parser.parse_args()
    verify(args.root, args.audit)
