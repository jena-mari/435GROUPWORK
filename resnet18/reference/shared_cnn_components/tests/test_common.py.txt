"""Tests for the shared ResNet-18 components. Run from the repository root:
    python -m unittest discover -s resnet18/tests -t . -v
Image-dependent tests are skipped until the original images are downloaded.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from resnet18 import common as cc

ROOT = cc.DATASET
HAVE_IMAGES = (ROOT / "White Background ").is_dir()


class TransformTests(unittest.TestCase):
    def setUp(self):
        a = np.full((416, 195, 3), 235, np.uint8)
        a[:, 70:125] = (40, 150, 40)
        self.portrait = Image.fromarray(a)
        self.landscape = self.portrait.transpose(Image.Transpose.ROTATE_90)

    def test_eval_sizes_and_determinism(self):
        for mode, (h, w) in cc.RESIZE_MODES.items():
            t = cc.EvalTransform(mode)
            x = t(self.portrait)
            self.assertEqual(tuple(x.shape), (3, h, w))
            self.assertTrue(torch.equal(x, t(self.portrait)))

    def test_landscape_turned_upright(self):
        self.assertEqual(cc.EvalTransform("portrait").pil(self.landscape).size, (224, 448))

    def test_training_augmentation_reproducible(self):
        a, b = cc.TrainTransform(11), cc.TrainTransform(11)
        xs = [a(self.portrait) for _ in range(3)]
        ys = [b(self.portrait) for _ in range(3)]
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(xs, ys)))
        self.assertFalse(torch.equal(xs[0], xs[1]))
        self.assertEqual(tuple(xs[0].shape), (3, 448, 224))


class ModelTests(unittest.TestCase):
    def test_batchnorm_statistics_fixed_after_train(self):
        model = cc.ResNet18()
        opt = torch.optim.Adam(cc.set_trainable(model, {"layer4": 1e-4, "fc": 1e-3}))
        before = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}
        model.train()
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)))
        loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 3, 448, 224)), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        opt.step()
        after = model.state_dict()
        self.assertTrue(all(torch.equal(v, after[k]) for k, v in before.items()))

    def test_trainable_counts(self):
        model = cc.ResNet18()
        cc.set_trainable(model, {"fc": 1e-3})
        self.assertEqual(cc.count_parameters(model)["trainable"], 512 * 5 + 5)
        cc.set_trainable(model, {"layer4": 1e-5, "fc": 1e-4})
        self.assertGreater(cc.count_parameters(model)["trainable"], 8_000_000)

    def test_checkpoint_round_trip(self):
        model = cc.ResNet18()
        with tempfile.TemporaryDirectory() as tmp:
            path = cc.save_checkpoint(Path(tmp) / "c.pt", model, method="test", seed=11)
            loaded, meta = cc.load_checkpoint(path)
        self.assertEqual(meta["seed"], 11)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values())))

    def test_selection_rule(self):
        t = cc.BestTracker(patience=2)
        self.assertEqual([t.update(f, l, e) for e, (f, l) in enumerate([(.5, 1), (.5, .9), (.4, 1), (.45, 1)], 1)],
                         [True, True, False, False])
        self.assertEqual(t.best_epoch, 2)
        self.assertTrue(t.should_stop)


@unittest.skipUnless(HAVE_IMAGES, "original images not downloaded")
class SharedSplitTests(unittest.TestCase):
    def test_seed_sizes(self):
        expected = {11: (777, 164, 165), 55: (777, 163, 166)}
        for seed, sizes in expected.items():
            data, _ = cc.load_shared_splits(ROOT, cc.AUDIT, seed)
            self.assertEqual(tuple(len(data[p]) for p in ("train", "validation", "test")), sizes)

    def test_prediction_export(self):
        data, _ = cc.load_shared_splits(ROOT, cc.AUDIT, 11)
        rows = [dict(image_id=r["image_id"], disease_id=int(r["disease_id"]), predicted_disease_id=0,
                     background_id=int(r["background_id"]), probs=np.full(5, .2)) for r in data["test"].rows]
        self.assertTrue(cc.check_predictions(rows, data["test"]))
        with self.assertRaises(ValueError):
            cc.check_predictions(rows[1:], data["test"])
        with tempfile.TemporaryDirectory() as tmp:
            cc.write_predictions(Path(tmp) / "p.csv", rows, 11, "frozen_resnet18")
            header = (Path(tmp) / "p.csv").read_text().splitlines()[0]
        self.assertEqual(header, "image_id,seed,method,disease_id,predicted_disease_id,background_id,"
                                 "prob_0,prob_1,prob_2,prob_3,prob_4")
        self.assertEqual(set(cc.get_scores(rows)), {"field", "white", "all"})


if __name__ == "__main__":
    unittest.main()
