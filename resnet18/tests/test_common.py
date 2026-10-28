import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_preprocessing.prepare_dataset import CLASSES, BACKGROUNDS, PARTITIONS, sha_file, write_csv, write_json
from resnet18.common import (
    BestCheckpoint, ImagePolicy, Letterbox, SharedData, assert_model_contract, assert_optimizer_contract,
    build_model, evaluate, export_predictions, image_transform, initialise_finetuned,
    load_checkpoint, make_optimizer, make_shared_data, prepare_training_mode,
    read_checkpoint, score_predictions, seed_everything, summarise_runs, train_one_epoch,
)
from resnet18.validate_implementation import check_changes, run_check, snapshot


class TinyImages(Dataset):
    def __init__(self, partition='train', count=5):
        self.rows = [dict(image_id=f'{partition}_image_{i}', partition=partition, seed=11,
                         disease_id=i % 5, background_id=i % 2,
                         disease=list(CLASSES)[i % 5], background=list(BACKGROUNDS)[i % 2]) for i in range(count)]
        self.images = torch.rand(count, 3, 32, 32, generator=torch.Generator().manual_seed(3))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return dict(image=self.images[i], image_id=self.rows[i]['image_id'],
                    disease_id=self.rows[i]['disease_id'], background_id=self.rows[i]['background_id'])


def context(scope='implementation_check'):
    return dict(seed=11, manifest_version='fixture', manifest_sha256={'train': 'a', 'validation': 'b', 'test': 'c'},
                policy=ImagePolicy().record(), class_mapping=CLASSES, background_mapping=BACKGROUNDS,
                run_scope=scope)


def selected_fixture_model():
    model = build_model(pretrained=False)
    # Isolate prediction-ID behaviour; real checkpoint round trips are tested separately.
    model.rice_checkpoint_context = context()
    model.rice_checkpoint_stage = 'frozen'
    return model


class CommonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_letterbox_retains_leaf_ends_and_evaluation_is_deterministic(self):
        im = Image.new('RGB', (20, 100), 'green')
        for x in range(20):
            im.putpixel((x, 0), (255, 0, 0)); im.putpixel((x, 99), (0, 0, 255))
        padded = Letterbox(224, (124, 116, 104))(im)
        self.assertEqual(padded.getpixel((112, 0)), (255, 0, 0))
        self.assertEqual(padded.getpixel((112, 223)), (0, 0, 255))
        transform = image_transform()
        self.assertEqual(tuple(transform(im).shape), (3, 224, 224))
        self.assertTrue(torch.equal(transform(im), transform(im)))

    def test_training_augmentation_can_be_repeated_with_the_seed(self):
        im = Image.fromarray(np.random.default_rng(3).integers(0, 256, (90, 30, 3), dtype=np.uint8))
        transform = image_transform(True)
        seed_everything(11); first = transform(im)
        seed_everything(11); second = transform(im)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

    def test_frozen_contract_rejects_train_mode_and_reapplies_batchnorm_policy(self):
        model = build_model(pretrained=False)
        self.assertEqual(assert_model_contract(model, 'frozen')['trainable_parameters'], 2565)
        model.train()
        with self.assertRaises(ValueError): assert_model_contract(model, 'frozen')
        prepare_training_mode(model, 'frozen')
        assert_model_contract(model, 'frozen')
        model.bn1.track_running_stats = False
        with self.assertRaisesRegex(ValueError, 'BatchNorm'): assert_model_contract(model, 'frozen')
        model.bn1.track_running_stats = True
        model.layer1[0].conv1.weight.requires_grad = True
        with self.assertRaisesRegex(ValueError, 'parameter'): assert_model_contract(model, 'frozen')

    def test_frozen_step_changes_only_the_head(self):
        model = build_model(pretrained=False)
        before = snapshot(model)
        optimizer = make_optimizer(model, 'frozen')
        result = train_one_epoch(model, DataLoader(TinyImages(count=2), batch_size=2), optimizer,
                                 torch.device('cpu'), 'frozen')
        self.assertEqual(result['images'], 2)
        check_changes(model, before, 'frozen')
        with self.assertRaisesRegex(ValueError, 'training partition'):
            train_one_epoch(model, DataLoader(TinyImages('validation'), batch_size=2), optimizer,
                            torch.device('cpu'), 'frozen')

    def test_finetuned_step_freezes_earlier_stages_and_all_batchnorm_buffers(self):
        model = build_model('finetuned', pretrained=False)
        contract = assert_model_contract(model, 'finetuned')
        self.assertEqual(contract['trainable_parameters'], 8396293)
        before = snapshot(model)
        optimizer = make_optimizer(model, 'finetuned')
        assert_optimizer_contract(model, optimizer, 'finetuned')
        train_one_epoch(model, DataLoader(TinyImages(count=2), batch_size=2), optimizer,
                        torch.device('cpu'), 'finetuned')
        check_changes(model, before, 'finetuned')
        optimizer.param_groups[0]['lr'] = 1e-3
        with self.assertRaisesRegex(ValueError, 'learning rates'):
            assert_optimizer_contract(model, optimizer, 'finetuned')

    def test_checkpoint_round_trip_identity_guards_and_finetuning_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = build_model(pretrained=False)
            optimizer = make_optimizer(model, 'frozen')
            path = Path(tmp) / 'best.pt'
            selector = BestCheckpoint(path, context(), 'frozen', patience=2)
            self.assertFalse(selector.update(.5, 1, model, optimizer))
            self.assertFalse(selector.update(.5, 2, model, optimizer))
            self.assertTrue(selector.update(.4, 3, model, optimizer))
            self.assertEqual(read_checkpoint(path, context(), 'frozen')['epoch'], 1)
            restored = build_model(pretrained=False)
            load_checkpoint(path, restored, context(), 'frozen')
            for key, value in model.state_dict().items(): self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            for key, value in [('seed', 22), ('manifest_version', 'other'), ('run_scope', 'experiment')]:
                wrong = copy.deepcopy(context()); wrong[key] = value
                with self.assertRaisesRegex(ValueError, 'mismatch'): read_checkpoint(path, wrong, 'frozen')
            wrong = copy.deepcopy(context()); wrong['policy']['size'] = 256
            with self.assertRaises(ValueError): read_checkpoint(path, wrong, 'frozen')
            with self.assertRaises(ValueError): read_checkpoint(path, context(), 'finetuned')
            fine, _ = initialise_finetuned(path, context(), torch.device('cpu'))
            self.assertEqual(fine.rice_frozen_checkpoint_sha256, sha_file(path))
            assert_model_contract(fine, 'finetuned')
            fine_selector = BestCheckpoint(Path(tmp) / 'fine.pt', context(), 'finetuned')
            fine_selector.update(.1, 1, fine, make_optimizer(fine, 'finetuned'))
            self.assertEqual(fine_selector.best_score, .1)

    def test_export_matches_ids_even_when_the_loader_is_shuffled(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = TinyImages('validation')
            loader = DataLoader(ds, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(9))
            rows = export_predictions(selected_fixture_model(), loader, Path(tmp) / 'predictions.csv',
                                      context(), 'frozen', torch.device('cpu'))
            self.assertEqual([r['image_id'] for r in rows], [r['image_id'] for r in ds.rows])
            self.assertTrue(all(abs(sum(r[f'prob_{i}'] for i in range(5)) - 1) < 1e-5 for r in rows))
            self.assertEqual(score_predictions(rows)['all']['images'], 5)
            with self.assertRaisesRegex(ValueError, 'must not export test'):
                export_predictions(build_model(pretrained=False), DataLoader(TinyImages('test'), batch_size=2),
                                   Path(tmp) / 'test.csv', context(), 'frozen', torch.device('cpu'))

    def test_export_detects_missing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Missing image'):
                export_predictions(selected_fixture_model(),
                    DataLoader(TinyImages('validation'), batch_size=2, drop_last=True),
                    Path(tmp) / 'predictions.csv', context(), 'frozen', torch.device('cpu'))

    def test_export_requires_the_selected_same_seed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = DataLoader(TinyImages('validation'), batch_size=2)
            with self.assertRaisesRegex(ValueError, 'selected checkpoint'):
                export_predictions(build_model(pretrained=False), loader, Path(tmp) / 'p.csv',
                                   context(), 'frozen', torch.device('cpu'))
            wrong = copy.deepcopy(context()); wrong['seed'] = 22
            with self.assertRaisesRegex(ValueError, 'loader seed'):
                export_predictions(selected_fixture_model(), loader, Path(tmp) / 'p.csv',
                                   wrong, 'frozen', torch.device('cpu'))

    def test_summaries_require_five_seeds_and_use_sample_std(self):
        scores = {seed: {bg: dict(accuracy=i / 10, macro_f1=i / 10) for bg in ('field', 'white', 'all')}
                  for i, seed in enumerate((11, 22, 33, 44, 55), 1)}
        row = summarise_runs(scores)[0]
        self.assertAlmostEqual(row['macro_f1_std'], np.std([.1, .2, .3, .4, .5], ddof=1))
        del scores[55]
        with self.assertRaisesRegex(ValueError, 'Five-run'): summarise_runs(scores)

    def test_invalid_offline_pretrained_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wrong.pth'; path.write_bytes(b'not the official weights')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                build_model(weights_path=path)
        with self.assertRaisesRegex(ValueError, 'same-split'):
            build_model('finetuned', pretrained=True)

    def test_two_stage_diagnostic_runner_creates_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = SharedData({part: TinyImages(part, count=10) for part in PARTITIONS}, {}, context())
            def untrained(stage='frozen', **_kwargs):
                return build_model(stage, pretrained=False)
            with patch('resnet18.validate_implementation.make_shared_data', return_value=data), \
                 patch('resnet18.validate_implementation.build_model', side_effect=untrained), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = run_check(root, root / 'audit', root / 'checks', requested_device='cpu')
            self.assertTrue(report['passed'])
            self.assertEqual(report['test_images_used'], 0)
            for stage in ('frozen', 'finetuned'):
                self.assertTrue((root / 'checks' / stage / 'run_config.json').exists())
                self.assertTrue((root / 'checks' / stage / 'validation_predictions.csv').exists())
                self.assertTrue(report['stages'][stage]['checkpoint_round_trip'])

    def test_shared_manifests_use_the_existing_mapping_and_ten_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit = root / 'preprocessing'; version = audit / 'fixture'
            (version / 'seed_11').mkdir(parents=True)
            (audit / 'CURRENT_SPLIT_VERSION.txt').write_text('fixture\n')
            hashes = {}
            for part in PARTITIONS:
                rows = []
                for disease, label in CLASSES.items():
                    for bg, bgid in BACKGROUNDS.items():
                        name = f'{part}_{label}_{bgid}'
                        path = root / f'{name}.png'; Image.new('RGB', (20, 50), (label * 40, bgid * 70, 30)).save(path)
                        # Use distinct source bytes across partitions for the audit fixture.
                        path.write_bytes(path.read_bytes() + part.encode())
                        rows.append(dict(path=path.name, image_id=name, disease=disease, disease_id=label,
                            background=bg, background_id=bgid, sha256=sha_file(path), group_id=name, seed=11, partition=part))
                file = version / 'seed_11' / f'{part}.csv'; write_csv(file, rows)
                hashes[f'seed_11/{part}.csv'] = sha_file(file)
            write_json(version / 'provenance.json', dict(eligible_images=30, manifest_sha256=hashes))
            data = make_shared_data(root, audit, 11, batch_size=3)
            self.assertEqual(next(iter(data.loaders['validation']))['image'].shape, (3, 3, 224, 224))
            self.assertEqual(len(data.datasets['train']), 10)
            self.assertEqual(data.context['manifest_sha256']['test'], hashes['seed_11/test.csv'])
            path = version / 'seed_11/train.csv'; path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum'): make_shared_data(root, audit, 11)


if __name__ == '__main__':
    unittest.main()
