import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from resnet18.check_cnn import check, main, preview
from resnet18.common import SharedData
from resnet18.tests.test_common import TinyImages, context


class SharedDiagnosticTests(unittest.TestCase):
    def test_preview_uses_training_images_and_preserves_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = TinyImages('train', count=10)
            for i, row in enumerate(dataset.rows):
                row['path'] = f'image_{i}.png'
                Image.new('RGB', (30, 100), (i * 20, 120, 30)).save(root / row['path'])
            data = SharedData({'train': dataset}, {}, context('experiment'))
            state = torch.random.get_rng_state()
            with patch('resnet18.check_cnn.cc.make_shared_data', return_value=data), contextlib.redirect_stdout(io.StringIO()):
                first, metadata = preview(root, root / 'audit', output=root / 'first')
                second, again = preview(root, root / 'audit', output=root / 'second')
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(metadata, again)
            self.assertEqual(metadata['source_partition'], 'train')
            self.assertEqual(metadata['test_images_evaluated'], 0)
            self.assertEqual(len(metadata['images']), 4)
            self.assertEqual({r['background'] for r in metadata['images']}, {'field', 'white'})
            self.assertTrue(all(r['image_id'].startswith('train_') for r in metadata['images']))
            with self.assertRaisesRegex(ValueError, 'fresh preview'):
                preview(root, root / 'audit', output=root / 'first')

    def test_consolidated_check_verifies_every_seed_and_uses_two_stage_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = {part: TinyImages(part, count=10) for part in ('train', 'validation', 'test')}
            def shared(_root, _audit, seed, **_kwargs):
                ctx = context(); ctx['seed'] = seed
                return SharedData(datasets, {}, ctx)
            def run(_root, _audit, output, **_kwargs):
                output.mkdir(parents=True)
                return dict(passed=True, test_images_used=0)
            with patch('resnet18.check_cnn.verify', return_value=(1106, 15)) as verification, \
                 patch('resnet18.check_cnn.cc.make_shared_data', side_effect=shared) as factory, \
                 patch('resnet18.check_cnn.run_check', side_effect=run) as runner:
                result = check(root, root / 'audit', root / 'checks', device='cpu')
            verification.assert_called_once()
            self.assertEqual(factory.call_count, 5)
            self.assertEqual(len(result['shared_splits']), 5)
            self.assertEqual(result['manifests_verified'], 15)
            self.assertEqual(result['test_images_used'], 0)
            self.assertEqual(runner.call_args.kwargs['requested_device'], 'cpu')
            self.assertTrue(json.loads((root / 'checks/shared_component_check.json').read_text())['passed'])

    def test_cli_accepts_shared_options_before_or_after_command(self):
        for argv in (['--dataset', '/tmp/rice', 'preview', '--seed', '22'],
                     ['preview', '--dataset', '/tmp/rice', '--seed', '22']):
            with patch('resnet18.check_cnn.preview') as command:
                main(argv)
            self.assertEqual(command.call_args.args[:3], (Path('/tmp/rice'),
                Path(__file__).resolve().parents[2] / 'Dhan-Shomadhan/preprocessing', 22))
        with patch('resnet18.train_frozen.main') as frozen:
            main(['smoke', '--dataset', '/tmp/rice', '--cpu'])
        self.assertIn('--smoke', frozen.call_args.args[0])
        self.assertIn('cpu', frozen.call_args.args[0])
