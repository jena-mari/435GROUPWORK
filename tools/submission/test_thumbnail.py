"""Small end-to-end feature extraction and training check for the portable control."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from evaluation.thumbnail_experiment import run


class ThumbnailTests(unittest.TestCase):
    def test_five_seed_control_exports_historical_schema_and_keeps_outputs_under_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            rows = []
            for i in range(10):
                path = root / f'image_{i}.png'
                Image.new('RGB', (30, 70), (i * 20, 80, 160)).save(path)
                rows.append(dict(path=path.name, image_id=f'id_{i}', disease_id=i % 5,
                                 background='field' if i < 5 else 'white'))
            splits = {part: SimpleNamespace(rows=rows) for part in ('train', 'validation', 'test')}
            with patch('evaluation.thumbnail_experiment.load_shared_splits', return_value=(splits, root)):
                run(root, root / 'audit', root / 'experiments/control', [11, 22, 33, 44, 55])
            with (root / 'experiments/control/test_predictions.csv').open() as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len(predictions), 50)
            self.assertEqual({int(r['seed']) for r in predictions}, {11, 22, 33, 44, 55})
            self.assertEqual(set(predictions[0]), {'seed', 'image_id', 'background', 'true_label', 'pred_label'})
            self.assertEqual(len(list((root / 'experiments/control/thumbnails').glob('*.jpg'))), 10)
            with self.assertRaises(ValueError):
                run(root, root / 'audit', root.parent / 'outside', [11])


if __name__ == '__main__':
    unittest.main()
