"""Check the shared-manifest integration and training-only scaler behaviour."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from hsv_lbp_svm import svm_baseline as svm
from data_preprocessing.prepare_dataset import read_csv, sha_file, write_csv, write_json


def fixture(root):
    audit = root / 'preprocessing'
    version = audit / 'conservative_splits' / 'fixture'
    manifests = version / 'seed_11'
    manifests.mkdir(parents=True)
    (audit / 'CURRENT_SPLIT_VERSION.txt').write_text('conservative_splits/fixture\n')
    hashes = {}
    for p, part in enumerate(svm.PARTITIONS):
        rows = []
        for disease, label in svm.CLASSES.items():
            for background, bgid in svm.BACKGROUNDS.items():
                ident = f'{part}_{label}_{bgid}'
                path = root / 'images' / f'{ident}.png'
                path.parent.mkdir(exist_ok=True)
                image = np.random.default_rng(p * 100 + label * 2 + bgid).integers(
                    0, 256, (30, 20, 3), dtype=np.uint8)
                Image.fromarray(image).save(path)
                rows.append(dict(path=path.relative_to(root).as_posix(), image_id=ident,
                    disease=disease, disease_id=label, background=background, background_id=bgid,
                    group_id=ident, sha256=sha_file(path), partition=part, seed=11))
        write_csv(manifests / f'{part}.csv', rows)
        hashes[f'seed_11/{part}.csv'] = sha_file(manifests / f'{part}.csv')
    write_json(version / 'provenance.json', dict(eligible_images=30, manifest_sha256=hashes))
    return audit, version


class SvmIntegrationTests(unittest.TestCase):
    def test_full_run_exports_the_shared_image_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit, version = fixture(root)
            output = root / 'experiments' / 'baseline'
            with contextlib.redirect_stdout(io.StringIO()):
                svm.run(root, audit, output, [11])
            rows = read_csv(output / 'seed_11' / 'test_predictions.csv')
            expected = read_csv(version / 'seed_11' / 'test.csv')
            self.assertEqual({r['image_id'] for r in rows}, {r['image_id'] for r in expected})
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r['method'] == 'baseline' and r['seed'] == '11' for r in rows))
            self.assertEqual(len(read_csv(output / 'seed_11' / 'validation_history.csv')), 16)
            scores = json.loads((output / 'seed_11' / 'metrics.json').read_text())
            self.assertEqual(scores['all']['images'], 10)
            self.assertEqual(scores['field']['images'], 5)
            self.assertEqual(np.asarray(scores['white']['confusion_matrix']).shape, (5, 5))
            summary = read_csv(output / 'summary_11.csv')
            self.assertTrue(all(r['runs'] == '1' and r['macro_f1_std'] == '' for r in summary))

    def test_validation_only_does_not_extract_or_score_test_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit, _ = fixture(root)
            output = root / 'experiments' / 'baseline_validation'
            with contextlib.redirect_stdout(io.StringIO()):
                svm.run(root, audit, output, [11], validation_only=True)
            cache = next((root / 'svm_cache').glob('*.npz'))
            saved = svm.load_saved_features(cache, svm.feature_signature())
            self.assertEqual(len(saved), 20)
            self.assertFalse((output / 'seed_11' / 'test_predictions.csv').exists())
            self.assertFalse((output / 'seed_11' / 'metrics.json').exists())

    def test_scaler_is_fit_on_training_features_only(self):
        X_train = np.arange(40).reshape(10, 4).astype(float)
        X_val = X_train + 10000
        y = np.tile(np.arange(5), 2)
        _, scaler, _, _, _ = svm.train_svm(X_train, y, X_val, y, 11)
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))
        self.assertEqual(scaler.n_samples_seen_, len(X_train))

    def test_cross_partition_groups_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit, version = fixture(root)
            train = read_csv(version / 'seed_11/train.csv')
            path = version / 'seed_11/test.csv'
            test = read_csv(path)
            test[0]['group_id'] = train[0]['group_id']
            write_csv(path, test)
            provenance_path = version / 'provenance.json'
            provenance = json.loads(provenance_path.read_text())
            provenance['manifest_sha256']['seed_11/test.csv'] = sha_file(path)
            write_json(provenance_path, provenance)
            with self.assertRaisesRegex(ValueError, 'crosses partitions'):
                svm.load_shared_splits(root, audit, 11)

    def test_changed_images_cannot_reuse_cached_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit, _ = fixture(root)
            datasets, _ = svm.load_shared_splits(root, audit, 11)
            ds = datasets['train']
            saved = {}
            svm.get_features(ds, saved)
            (root / ds.rows[0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed since audit'):
                svm.get_features(ds, saved)

    def test_summary_uses_sample_standard_deviation(self):
        results = [{s: dict(accuracy=v, macro_f1=v) for s in ('field', 'white', 'all')}
                   for v in (.2, .6)]
        row = svm.summary_rows(results)[0]
        self.assertAlmostEqual(row['accuracy_mean'], .4)
        self.assertAlmostEqual(row['accuracy_std'], np.std([.2, .6], ddof=1))


if __name__ == '__main__':
    unittest.main()
