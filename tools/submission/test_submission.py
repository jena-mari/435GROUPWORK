"""Tests for standalone packaging, preserved audit data and experiment routing."""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

DEFAULT_SCRIPT = Path(__file__).with_name('group_name.py')
if not DEFAULT_SCRIPT.is_file():
    DEFAULT_SCRIPT = Path(__file__).resolve().parents[2] / 'CSCI435Group2/group_name.py'
SCRIPT = Path(os.environ.get('SUBMISSION_SCRIPT', DEFAULT_SCRIPT))
spec = importlib.util.spec_from_file_location('submission', SCRIPT)
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        temp = patch.object(tempfile, 'tempdir', tempfile.gettempdir())
        temp.start()
        self.addCleanup(temp.stop)

    def test_bundled_repository_source_hashes_and_all_experiment_commands(self):
        files = submission._bundle()['modules']
        manifest = json.loads(base64.b64decode(files['source_manifest.json']))
        for name, sha in manifest.items():
            self.assertEqual(hashlib.sha256(base64.b64decode(files[name])).hexdigest(), sha)
        for command in ('train', 'train-svm', 'train-frozen', 'train-finetuned',
                        'train-cnns', 'train-thumbnail', 'compare-resize', 'smoke',
                        'evaluate-reported', 'evaluate-thumbnail'):
            self.assertIn(command, submission.COMMANDS)
        self.assertNotIn('.jpg', '\n'.join(files))

    def test_shared_audit_is_byte_exact_idempotent_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            audit = root / 'preprocessing'
            runtime = submission._materialise(root, audit)
            original = (audit / 'inventory.csv').read_bytes()
            self.assertEqual(submission._materialise(root, audit), runtime)
            summary = json.loads((audit / 'audit_summary.json').read_text())
            self.assertEqual(hashlib.sha256(original).hexdigest(), summary['inventory_sha256'])
            (audit / 'inventory.csv').write_bytes(b'user modification')
            with self.assertRaisesRegex(ValueError, 'preserved unchanged'):
                submission._materialise(root, audit)
            self.assertEqual((audit / 'inventory.csv').read_bytes(), b'user modification')

    def test_no_output_or_bundled_path_can_escape_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(ValueError):
                submission._inside(root.parent / 'outside', root)
            with self.assertRaises(ValueError):
                submission._write_bundle({'../outside': 'YQ=='}, root, immutable=True)

    def test_new_runs_are_unique_and_separate_stages_reuse_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(dataset=Path(tmp).resolve(), output=None)
            first = submission._training_output(args, fresh=True)
            self.assertEqual(submission._training_output(args), first)
            second = submission._training_output(args, fresh=True)
            self.assertNotEqual(first, second)
            self.assertTrue(second.is_relative_to(args.dataset))

    def test_combined_training_dispatches_to_two_environments_and_all_five_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / 'pretrained').mkdir()
            (root / 'pretrained/resnet18-f37072fd.pth').write_bytes(b'fixture')
            with patch.object(submission, '_verify'), patch.object(submission, '_python', return_value='python'), \
                    patch.object(submission, '_call') as call:
                submission.main(['train', '--dataset', str(root)])
            self.assertEqual([c.args[0] for c in call.call_args_list], ['svm', 'cnn'])
            self.assertEqual([c.args[1] for c in call.call_args_list],
                             ['hsv_lbp_svm.svm_baseline', 'resnet18.run_role6'])
            for invocation in call.call_args_list:
                args = invocation.args[2]
                self.assertEqual(args[args.index('--seeds') + 1:], list(submission.SEEDS))

    def test_report_evaluation_needs_no_images_and_allocates_a_new_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(submission, '_call') as call:
                submission.main(['evaluate-reported', '--dataset', str(root)])
                first = call.call_args.args[2]
                submission.main(['evaluate-reported', '--dataset', str(root)])
                second = call.call_args.args[2]
            self.assertNotEqual(first[first.index('--out')+1], second[second.index('--out')+1])
            self.assertFalse((root / 'Field Background  ').exists())


if __name__ == '__main__':
    unittest.main()
