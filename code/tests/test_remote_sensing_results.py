import json
from pathlib import Path
import tempfile
import unittest

from scripts.remote_sensing_results import write_portable_result


class PortableResultsTest(unittest.TestCase):
    def test_matching_source_and_tensor_fields_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "source.json"
            before = {"miou": 0.494309839901638, "valid_pixels": 682354901,
                      "iou_by_class": {"forest": 0.4883603580905165}}
            after = {"miou": 0.4111236179528476, "valid_pixels": 682354901,
                     "iou_by_class": {"forest": 0.1669942922206072}}
            baseline.write_text(json.dumps({"task": "r2u", "split": "loveda_urban_val",
                                           "source_checkpoint_sha256": "same", "metrics": before}))
            model = {"stage": "adapt", "iteration": 30000,
                     "training_settings": {"relation_weight": 0},
                     "source_checkpoint_sha256": "same", "feature_extractor": object(),
                     "optimizer": object()}
            path = root / "result_portable.json"
            write_portable_result(path, task="r2u", split="loveda_urban_val", metrics=after,
                                  model=model, protocol={}, source_baseline_path=baseline)
            report = json.loads(path.read_text())
            self.assertEqual(report["method"], "pseudo_label_baseline")
            self.assertFalse(report["contains_weights"])
            self.assertNotIn("optimizer", report["model"])
            self.assertNotIn("feature_extractor", report["model"])
            self.assertAlmostEqual(report["metrics_percent"]["miou"], 41.11236179528476)
            self.assertAlmostEqual(report["source_comparison"]["delta_miou_percentage_points"],
                                   -8.31862219487904)
            self.assertAlmostEqual(report["source_comparison"]["delta_iou_by_class_percentage_points"]["forest"],
                                   -32.13660658699093)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_mismatched_checkpoint_never_receives_a_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "source.json"
            metrics = {"miou": 0.4, "valid_pixels": 10, "iou_by_class": {"car": 0.4}}
            baseline.write_text(json.dumps({"task": "vh2pd", "split": "isprs_pd_rgb_eval",
                                           "source_checkpoint_sha256": "other", "metrics": metrics}))
            report = write_portable_result(root / "result.json", task="vh2pd", split="isprs_pd_rgb_eval",
                                          metrics=metrics, protocol={}, source_baseline_path=baseline,
                                          model={"stage": "adapt", "source_checkpoint_sha256": "current"})
            self.assertEqual(report["source_comparison"], {"status": "source_checkpoint_mismatch"})

    def test_eval_export_uses_saved_method_and_preserves_training_protocol(self):
        # Real CPU evaluator on four synthetic pixels; no training or large weights.
        import sys
        from unittest.mock import patch
        import torch
        from torch import nn
        import torch.nn.functional as functional
        import train_SRMP_remote as trainer

        class Encoder(nn.Module):
            def forward(self, images):
                return images, images

        class Classifier(nn.Module):
            def forward(self, features, size=None):
                return functional.one_hot(features[:, 0].long(), 7).permute(0, 3, 1, 2).float() * 10

        class Dataset(torch.utils.data.Dataset):
            classes, role = 7, "eval"

            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, index):
                images = torch.zeros(3, 2, 2)
                images[0] = torch.tensor([[0, 6], [6, 2]])
                return images, torch.tensor([[0, 1], [6, 255]])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            (data / "manifests").mkdir(parents=True)
            (data / "manifests/loveda_urban_val.csv").write_text("image,label\n")
            (data / "metadata.json").write_text("{}")
            checkpoint_path = root / "srmp_last.pth"
            checkpoint_path.write_bytes(b"small synthetic checkpoint")
            origin = {"files_sha256": {"trainer": "original-training-code"}, "args": {"steps": 30000}}
            origin_path = root / "protocol_adapt.json"
            origin_path.write_text(json.dumps(origin))
            checkpoint = {"task": "r2u", "num_classes": 7, "stage": "adapt", "iteration": 30000,
                          "feature_extractor": {}, "classifier": {}, "source_checkpoint_sha256": "original-source",
                          "training_settings": {"relation_weight": 0}, "optimizer": torch.ones(3)}
            argv = ["train_SRMP_remote.py", "--task", "r2u", "--stage", "eval", "--workers", "0",
                    "--data-root", str(data), "--source-checkpoint", str(checkpoint_path), "--output-dir", str(root)]
            with patch.object(sys, "argv", argv), patch.object(trainer, "make_model", return_value=(Encoder(), Classifier())), \
                    patch.object(trainer, "ManifestDataset", Dataset), patch.object(trainer, "load_checkpoint", return_value=checkpoint), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                trainer.main()
            report = json.loads((root / "result_portable.json").read_text())
            self.assertEqual(report["method"], "pseudo_label_baseline")  # eval CLI defaults to relation_weight=0.1
            self.assertEqual(report["model"]["iteration"], 30000)
            self.assertEqual(report["model"]["inference_model"], "ema_teacher")
            self.assertEqual(report["training_protocol"], origin)
            self.assertEqual(report["evaluation_protocol"]["split"], "loveda_urban_val")
            self.assertEqual(report["evaluation_protocol"]["stage"], "eval")
            self.assertIn("evaluated_checkpoint", report["evaluation_protocol"]["files_sha256"])
            self.assertEqual(json.loads(origin_path.read_text()), origin)
            self.assertEqual(report["metrics"]["valid_pixels"], 3)
            self.assertAlmostEqual(report["metrics"]["miou"], 0.5)
            matrix = report["metrics"]["confusion_matrix"]
            self.assertEqual(matrix[1][6], 1)
            self.assertEqual(sum(map(sum, matrix)), 3)
            self.assertNotIn("optimizer", report["model"])


if __name__ == "__main__":
    unittest.main()
