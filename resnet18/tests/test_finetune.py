"""Person 6 safeguards, gradients, and validation/test isolation."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from resnet18 import common as cc, finetune as ft


class FineTuneTests(unittest.TestCase):
    def test_only_layer4_and_head_change_and_batchnorm_stays_fixed(self):
        cc.set_seed(11)
        model = cc.ResNet18()
        opt = torch.optim.Adam(cc.set_trainable(model, {"layer4": 1e-3, "fc": 1e-3}))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batch = {"image": torch.randn(2, 3, 32, 32), "disease_id": torch.tensor([0, 1])}
        loss, _ = ft.train_epoch(model, [batch], opt, "cpu")
        self.assertTrue(loss > 0)
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
        self.assertTrue(any(k.startswith("layer4.") for k in changed))
        self.assertTrue(any(k.startswith("fc.") for k in changed))
        self.assertTrue(all(k.startswith(("layer4.", "fc.")) for k in changed))
        self.assertFalse(any("running_" in k or "num_batches_tracked" in k for k in changed))

    def test_person5_format_and_mismatched_seed_rejected(self):
        data, version = cc.load_shared_splits(seed=11)
        cfg = cc.base_run_config("frozen_resnet18", 11, version, data, "cpu", trainable_stages={"fc": 1e-3})
        cfg["transforms"] = ft.transform_record(data)
        ft.validate_frozen(cfg, 11, version, data, cc.AUDIT)
        for key, bad in (("seed", 22), ("common_code_sha256", "wrong"),
                         ("trainable_stages", {"layer4": 1e-5}), ("batchnorm", "updating")):
            altered = copy.deepcopy(cfg)
            altered[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                ft.validate_frozen(altered, 11, version, data, cc.AUDIT)

    def test_full_selection_never_predicts_test_partition(self):
        data, version = cc.load_shared_splits(seed=11)
        val_rows = [dict(image_id=r["image_id"], disease_id=int(r["disease_id"]),
                         background_id=int(r["background_id"]), predicted_disease_id=0,
                         probs=[.2] * 5) for r in data["validation"].rows]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen" / "seed_11"
            frozen.mkdir(parents=True)
            cfg = cc.base_run_config("frozen_resnet18", 11, version, data, "cpu", trainable_stages={"fc": 1e-3})
            cc.write_json(frozen / "run_config.json", cfg)
            cc.save_checkpoint(frozen / "best_checkpoint.pt", cc.ResNet18(), seed=11, method="frozen_resnet18")
            args = SimpleNamespace(root=root, frozen_dir=root / "frozen", audit=cc.AUDIT,
                                   checkpoint_name="best_checkpoint.pt", workers=0, epochs=1, cpu=True)
            def prediction(model, loader, device):
                self.assertIs(loader, data["validation"])
                return dict(rows=val_rows, loss=1.0, accuracy=.2, macro_f1=.1)
            with patch.object(cc, "load_shared_splits", return_value=(data, version)), \
                 patch.object(cc, "make_loader", side_effect=lambda ds, *a, **kw: ds), \
                 patch.object(ft, "train_epoch", return_value=(1., .2)), \
                 patch.object(cc, "predict", side_effect=prediction):
                ft.select_seed(args, 11)
            out = root / "experiments" / ft.METHOD / "seed_11"
            self.assertFalse((out / "test_predictions.csv").exists())
            record = ft.read_json(out / "run_config.json")
            self.assertEqual(record["status"], "selected_validation_only")
            self.assertEqual(record["selected"]["candidate"], 0)
            self.assertEqual(len(record["candidates"]), 2)
            test_rows = [dict(image_id=r["image_id"], disease_id=int(r["disease_id"]),
                              background_id=int(r["background_id"]), predicted_disease_id=0,
                              probs=[.2] * 5) for r in data["test"].rows]
            with patch.object(cc, "load_shared_splits", return_value=(data, version)), \
                 patch.object(cc, "make_loader", side_effect=lambda ds, *a, **kw: ds), \
                 patch.object(cc, "predict", return_value=dict(rows=test_rows)) as predictor:
                scores = ft.evaluate_seed(args, 11)
                self.assertEqual(set(scores), {"field", "white", "all"})
                self.assertEqual(predictor.call_count, 1)
                with self.assertRaises(ValueError):
                    ft.evaluate_seed(args, 11)
                self.assertEqual(predictor.call_count, 1)
            self.assertEqual(ft.read_json(out / "run_config.json")["status"], "test_evaluated")
            self.assertTrue((out / "test_predictions.csv").is_file())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
