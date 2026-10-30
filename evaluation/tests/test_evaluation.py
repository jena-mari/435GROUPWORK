import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_preprocessing.prepare_dataset import CLASSES, BACKGROUNDS, SEEDS, sha_file, write_csv, write_json
from evaluation.role4_evaluation import evaluate_sources, normalise_predictions, summarise


def fixture(root):
    audit, sources = root / 'audit', {'baseline': root / 'baseline', 'frozen_resnet18': root / 'frozen'}
    version = audit / 'v1'
    version.mkdir(parents=True)
    (audit / 'CURRENT_SPLIT_VERSION.txt').write_text('v1\n')
    hashes = {}
    for index, seed in enumerate(SEEDS):
        directory = version / f'seed_{seed}'
        directory.mkdir()
        manifest = [dict(image_id=f'image_{i}', seed=seed, partition='test', disease_id=i % 5,
                         background_id=i // 5, path=f'image_{i}.jpg') for i in range(10)]
        local_hashes = {}
        for part in ('train', 'validation', 'test'):
            file = directory / f'{part}.csv'
            write_csv(file, [dict(row, partition=part) for row in manifest])
            hashes[f'seed_{seed}/{part}.csv'] = sha_file(file)
            local_hashes[part] = sha_file(file)
        for method, output in sources.items():
            run = output / f'seed_{seed}'
            run.mkdir(parents=True)
            rows = [dict(image_id=row['image_id'], seed=seed, method=method,
                         disease_id=row['disease_id'], background_id=row['background_id'],
                         predicted_disease_id=row['disease_id'] if i >= index else (row['disease_id'] + 1) % 5)
                    for i, row in enumerate(manifest)]
            write_csv(run / 'test_predictions.csv', rows)
            context = dict(seed=seed, manifest_version='v1', manifest_sha256=local_hashes,
                           class_mapping=CLASSES, background_mapping=BACKGROUNDS,
                           run_scope='experiment', policy={'size': 224})
            write_json(run / 'run_config.json', dict(method=method, context=context, weights_source='IMAGENET1K_V1'))
    write_json(version / 'provenance.json', dict(manifest_sha256=hashes))
    return audit, sources


class EvaluationTests(unittest.TestCase):
    def test_five_seed_comparison_uses_sample_std_and_combined_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit, sources = fixture(root)
            summary = evaluate_sources(sources, audit, root / 'out', plots=False)
            row = next(r for r in summary if r['method'] == 'baseline' and r['scenario'] == 'all')
            self.assertAlmostEqual(row['accuracy_mean'], .8)
            self.assertAlmostEqual(row['accuracy_std'], np.std([1, .9, .8, .7, .6], ddof=1))
            self.assertTrue((root / 'out/paired_differences.csv').exists())
            self.assertTrue((root / 'out/per_class_metrics.csv').exists())
            config = json.loads((root / 'out/evaluation_config.json').read_text())
            self.assertFalse(config['final_group_comparison'])
            with self.assertRaisesRegex(ValueError, 'exists'):
                evaluate_sources(sources, audit, root / 'out', plots=False)

    def test_missing_run_is_rejected_before_writing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit, sources = fixture(root)
            (sources['baseline'] / 'seed_55/test_predictions.csv').unlink()
            with self.assertRaises(FileNotFoundError): evaluate_sources(sources, audit, root / 'out', plots=False)
            self.assertFalse((root / 'out').exists())

    def test_wrong_background_seed_method_or_duplicate_id_is_rejected(self):
        expected = {'a': dict(disease_id=0, background_id=0, path='a.jpg')}
        original = dict(image_id='a', seed=11, method='baseline', disease_id=0,
                        predicted_disease_id=0, background_id=0)
        for changes in ({'background_id': 1}, {'seed': 22}, {'method': 'other'}, {'disease_id': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalise_predictions([dict(original, **changes)], expected, 11, 'baseline')
        with self.assertRaises(ValueError): normalise_predictions([original, original], expected, 11, 'baseline')
        with self.assertRaises(ValueError): normalise_predictions([], expected, 11, 'baseline')

    def test_mismatched_preprocessing_or_diagnostic_scope_is_rejected(self):
        for change in ('policy', 'scope', 'manifest'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); audit, sources = fixture(root)
                path = sources['frozen_resnet18'] / 'seed_55/run_config.json'
                config = json.loads(path.read_text())
                if change == 'policy': config['context']['policy']['size'] = 448
                elif change == 'scope': config['context']['run_scope'] = 'implementation_check'
                else: config['context']['manifest_sha256']['test'] = 'changed'
                write_json(path, config)
                with self.assertRaises(ValueError): evaluate_sources(sources, audit, root / 'out', plots=False)
                self.assertFalse((root / 'out').exists())

    def test_partial_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Five distinct'):
            summarise([dict(method='baseline', scenario='white', seed=11, accuracy=1)])

    def test_thumbnail_reference_is_separate_and_checked_against_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit, sources = fixture(root)
            reference = root / 'reference'; reference.mkdir()
            rows = [dict(image_id=f'image_{i}', seed=seed, background='field' if i < 5 else 'white',
                         true_label=i % 5, pred_label=i % 5) for seed in SEEDS for i in range(10)]
            write_csv(reference / 'test_predictions.csv', rows)
            result = evaluate_sources({}, audit, root / 'out', thumbnail_reference=reference, plots=False)
            self.assertEqual(result[0]['method'], 'thumbnail_svm_preliminary')
            with self.assertRaises(ValueError):
                evaluate_sources(sources, audit, root / 'mixed', thumbnail_reference=reference, plots=False)
