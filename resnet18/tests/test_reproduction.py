"""Output routing and resume checks for the five-seed training workflow."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from resnet18 import common as cc, run_role6 as runner


class ReproductionTests(unittest.TestCase):
    def test_output_directory_is_separate_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run = cc.new_run_dir(root / 'data', 'frozen_resnet18', 11, output_root=root / 'new')
            self.assertEqual(run, root / 'new/frozen_resnet18/seed_11')
            self.assertFalse((root / 'data/experiments').exists())
            with self.assertRaises(ValueError):
                cc.new_run_dir(root / 'data', 'frozen_resnet18', 11, output_root=root / 'new')

    def test_config_without_checkpoint_does_not_silently_skip_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run = root / 'frozen_resnet18/seed_11'
            run.mkdir(parents=True)
            (run / 'run_config.json').write_text('{}')
            with patch('sys.argv', ['run', '--output', str(root), '--seeds', '11']), \
                    contextlib.redirect_stderr(io.StringIO()) as error, \
                    patch.object(runner.train_frozen, 'run_seed') as train:
                with self.assertRaises(SystemExit) as exc:
                    runner.main()
                self.assertEqual(exc.exception.code, 2)
                self.assertIn('fresh directory', error.getvalue())
                train.assert_not_called()

    def test_all_stages_use_new_output_and_selection_precedes_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / 'new'
            events = []

            def frozen(seed, args, device):
                self.assertEqual(args.dataset, root / 'data')
                self.assertEqual(args.output, output)
                run = cc.new_run_dir(args.dataset, 'frozen_resnet18', seed, output_root=args.output)
                for name in ('run_config.json', 'best_checkpoint.pt', 'metrics.json', 'test_predictions.csv'):
                    (run / name).write_text('{}')
                events.append(('frozen', seed))

            def select(args, seed):
                self.assertEqual(args.frozen_dir, output / 'frozen_resnet18')
                self.assertEqual(args.output, output)
                run = cc.new_run_dir(args.root, 'finetuned_resnet18', seed, output_root=args.output)
                cc.write_json(run / 'run_config.json', dict(status='selected_validation_only'))
                events.append(('select', seed))

            def evaluate(args, seed):
                self.assertEqual(args.output, output)
                events.append(('evaluate', seed))
                return {s: dict(accuracy=.5, macro_f1=.4) for s in ('field', 'white', 'all')}

            with patch('sys.argv', ['run', '--dataset', str(root / 'data'), '--output', str(output),
                                    '--workers', '0', '--cpu', '--seeds', '11', '22']), \
                    patch.object(runner.train_frozen, 'run_seed', side_effect=frozen), \
                    patch.object(runner.finetune, 'select_seed', side_effect=select), \
                    patch.object(runner.finetune, 'evaluate_seed', side_effect=evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(events, [('frozen', 11), ('frozen', 22), ('select', 11),
                                      ('select', 22), ('evaluate', 11), ('evaluate', 22)])
            self.assertTrue((output / 'finetuned_resnet18/summary_seeds_11_22.csv').is_file())
            self.assertFalse((root / 'data/experiments').exists())


if __name__ == '__main__':
    unittest.main()
