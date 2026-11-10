import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

import train_CROTS_whitebox as whitebox
from scripts.diagnose_crots_remote import EvaluationImages, evaluate_at_size, summarize_matrix
from scripts.crots_algorithm import smhm_loss


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
    def forward(self, x):
        y = self.conv(x)
        return y, y


class Classifier(nn.Conv2d):
    def __init__(self, classes):
        super().__init__(4, classes, 1)
    def forward(self, x, size=None):
        y = super().forward(x)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=True) if size else y


class WhiteboxTests(unittest.TestCase):
    def test_source_weights_teacher_and_resume_for_both_tasks(self):
        for task, classes in (("vh2pd", 6), ("r2u", 7)):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = []
                for i in range(2):
                    Image.fromarray(np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root/f"{i}.png")
                    rows.append({"image": f"{i}.png", "label": ""})
                args = SimpleNamespace(output=root/"run", data_root=root, source_checkpoint=root/"source.pth",
                                       seed=1, workers=0, smoke=True, task=task)
                config = json.loads((whitebox.ROOT/f"configs/crots_whitebox_{task}.json").read_text())
                config.update(image_size=32, batch_size=2, adapt_epochs_per_round=2, rounds=1)
                args.output.mkdir()
                snapshots = []
                captured = []
                original_save = whitebox.write_checkpoint
                def save(path, payload):
                    original_save(path, payload)
                    if payload["completed_epochs"] == 1:
                        snapshots.append(path.read_bytes())
                with patch.object(whitebox, "make_model", side_effect=lambda n: (Encoder(), Classifier(n))):
                    torch.manual_seed(9)
                    source = whitebox.Segmenter(classes)
                    expected = copy.deepcopy(source.state_dict())
                    torch.save({"task": task, "stage": "source", "num_classes": classes,
                                "normalization": "imagenet_rgb_planes", "source_split": whitebox.TASKS[task][1],
                                "feature_extractor": source.encoder.state_dict(), "classifier": source.classifier.state_dict()}, args.source_checkpoint)
                    protocol = {"signature": "test-whitebox"}
                    device = torch.device("cpu")
                    whitebox.prepare(args, config, rows, device, protocol)
                    def capture(student, teacher, *values, **kw):
                        if not captured:
                            for name, value in expected.items():
                                torch.testing.assert_close(student.state_dict()[name], value, rtol=0, atol=0)
                                torch.testing.assert_close(teacher.state_dict()[name], value, rtol=0, atol=0)
                            captured.append(True)
                        return smhm_loss(student, teacher, *values, **kw)
                    with patch.object(whitebox, "smhm_loss", side_effect=capture), patch.object(whitebox, "write_checkpoint", side_effect=save):
                        whitebox.train_stage(args, config, rows, device, protocol, 1)
                    self.assertTrue(captured)
                    self.assertFalse((args.output/"distill").exists())
                    last = args.output/"adapt_round1/last.pth"
                    expected_final = whitebox.load_checkpoint(last)
                    last.write_bytes(snapshots[0])
                    whitebox.train_stage(args, config, rows, device, protocol, 1)
                    actual = whitebox.load_checkpoint(last)
                    for name in ("student", "teacher"):
                        for key in actual[name]:
                            torch.testing.assert_close(actual[name][key], expected_final[name][key], rtol=0, atol=0)
                    digest = whitebox.sha256_file(last)
                    whitebox.train_stage(args, config, rows, device, protocol, 1)
                    self.assertEqual(digest, whitebox.sha256_file(last))
                    with self.assertRaisesRegex(ValueError, "no ImageNet"):
                        whitebox.train_stage(args, config, rows, device, protocol, 0)

    def test_siamseg_metrics_and_five_class_mean(self):
        matrix = np.eye(6, dtype=np.int64)
        matrix[5, 5] = 0
        matrix[5, 4] = 1
        metrics = summarize_matrix(matrix)
        self.assertAlmostEqual(metrics['miou'], .75)
        self.assertAlmostEqual(metrics['miou5'], .9)
        self.assertAlmostEqual(metrics['overall_accuracy'], 5/6)
        self.assertAlmostEqual(metrics['mean_accuracy'], 5/6)
        self.assertAlmostEqual(metrics['precision_by_class'][metrics['class_order'][4]], .5)
        self.assertAlmostEqual(metrics['f1_by_class'][metrics['class_order'][4]], 2/3)

    def test_scale_diagnostic_keeps_native_gt_and_ignore_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"manifests").mkdir()
            (root/"manifests/test.csv").write_text("image,label\nimage.png,mask.png\n")
            Image.new("RGB", (12, 10)).save(root/"image.png")
            mask = np.full((10, 12), 2, dtype=np.uint8)
            mask[0] = 255
            Image.fromarray(mask).save(root/"mask.png")
            dataset = EvaluationImages(root, "test", 7, 4)
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 4, 4))
            self.assertEqual(tuple(label.shape), (10, 12))
            class ConstantModel(nn.Module):
                def forward(self, x, size):
                    result = torch.zeros((len(x), 7, *size))
                    result[:, 2] = 1
                    return result
            metrics = evaluate_at_size(ConstantModel(), dataset, 7, torch.device("cpu"), 0, False)
            self.assertEqual(metrics["valid_pixels"], 108)
            self.assertEqual(metrics["iou_by_class"]["road"], 1)
            self.assertEqual(metrics["confusion_matrix"][2][2], 108)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
