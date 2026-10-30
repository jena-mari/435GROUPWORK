"""Exercise the supplied experiment loop against real shared checkpoint helpers."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from resnet18.common import SharedData, build_model, initialise_finetuned
from resnet18.train_frozen import run_seed
from resnet18.tests.test_common import TinyImages, context


class FrozenExperimentTests(unittest.TestCase):
    def test_selected_checkpoint_exports_and_same_split_finetuning(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                datasets = {p: TinyImages(p, 10) for p in ('train', 'validation', 'test')}
                data = SharedData(datasets, {p: DataLoader(ds, batch_size=4) for p, ds in datasets.items()},
                                  context('experiment'))
                args = argparse.Namespace(dataset=root, audit=root / 'audit', output=root / 'runs',
                    batch_size=4, workers=0, smoke=False, evaluate_only=False, weights_path=None,
                    patience=5, epochs=1, validation_only=True)
                def fixture_model(*_args, **_kwargs):
                    model = build_model(pretrained=False)
                    model.rice_weights_source = 'IMAGENET1K_V1'  # No download in offline fixture.
                    return model
                with patch('resnet18.train_frozen.cc.make_shared_data', return_value=data), \
                     patch('resnet18.train_frozen.cc.build_model', side_effect=fixture_model), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertIsNone(run_seed(11, args, torch.device('cpu')))
                directory = root / 'runs/seed_11'
                self.assertFalse((directory / 'test_predictions.csv').exists())
                self.assertTrue(json.loads((directory / 'run_config.json').read_text())['training_complete'])
                fine, source = initialise_finetuned(directory / 'best.pt', data.context, torch.device('cpu'))
                self.assertEqual(source['epoch'], 1)
                self.assertTrue(fine.fc.weight.requires_grad)
                args.evaluate_only, args.validation_only = True, False
                with patch('resnet18.train_frozen.cc.make_shared_data', return_value=data), contextlib.redirect_stdout(io.StringIO()):
                    scores = run_seed(11, args, torch.device('cpu'))
                self.assertEqual(scores['all']['images'], 10)
                self.assertTrue((directory / 'test_predictions.metadata.json').exists())
        finally:
            torch.set_num_threads(previous)

    def test_smoke_check_cannot_export_test_predictions(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                datasets = {p: TinyImages(p, 10) for p in ('train', 'validation', 'test')}
                data = SharedData(datasets, {p: DataLoader(ds, batch_size=4) for p, ds in datasets.items()}, context())
                args = argparse.Namespace(dataset=root, audit=root / 'audit', output=root / 'smoke',
                    batch_size=4, workers=0, smoke=True, evaluate_only=False, weights_path=None,
                    patience=5, epochs=1, validation_only=True)
                fixture_model = build_model(pretrained=False)
                with patch('resnet18.train_frozen.cc.make_shared_data', return_value=data), \
                     patch('resnet18.train_frozen.cc.build_model', return_value=fixture_model), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run_seed(11, args, torch.device('cpu'))
                config = json.loads((root / 'smoke/seed_11/run_config.json').read_text())
                self.assertEqual(config['context']['run_scope'], 'implementation_check')
                self.assertFalse((root / 'smoke/seed_11/test_predictions.csv').exists())
        finally:
            torch.set_num_threads(previous)
