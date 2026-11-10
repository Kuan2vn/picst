import ast
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from scripts.crots_algorithm import smhm_loss, augment_rgb
import train_CROTS_remote as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/crots_r2u.json").read_text())


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
    def forward(self, x):
        out = self.conv(x)
        return out, out


class TinyClassifier(nn.Conv2d):
    def __init__(self):
        super().__init__(4, 7, 1)
    def forward(self, x, size=None):
        return super().forward(x)


class TinyNet(nn.Conv2d):
    def __init__(self):
        super().__init__(3, 7, 1)
    def forward(self, x, size=None):
        return super().forward(x)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class CROTSParityTests(unittest.TestCase):
    def test_smhm_loss_gradients_and_ema_match_pinned_author_code(self):
        runner.reference_checks()
        tree = ast.parse((ROOT / "references/crots/modules/smhm.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SMHMModule")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "training_step")
        noop = lambda **kw: (kw["data"], kw["target"])
        namespace = {"torch": torch, "np": np, "random": random, "F": F,
                     "transformsgpu": SimpleNamespace(color_jitter=noop, gaussian_blur=noop)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "pinned_crots_smhm", "exec"), namespace)
        seed(12)
        student, teacher = TinyNet(), TinyNet().requires_grad_(False)
        other_student, other_teacher = copy.deepcopy(student), copy.deepcopy(teacher)
        images, patches = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 2, 2)
        labels = torch.randint(7, (4, 2, 2))
        class Reference:
            net, ema_net = student, teacher
            params = SimpleNamespace(position_beta=0.25, visualize_interval=0)
            global_step, local_rank, device = 5, 0, torch.device("cpu")
            image_mean = torch.zeros(3)
            def __call__(self, *args, **kwargs): return self.net(*args, **kwargs)
            def safe_iter_patch(self): return {"image": patches, "label": labels}
            def log_dict(self, *args, **kwargs): pass
        seed(101)
        expected = namespace["training_step"](Reference(), {"image": images}, 0)
        expected.backward()
        def no_augmentation(value):
            np.random.uniform()
            np.random.uniform()
            return value
        seed(101)
        actual, _ = smhm_loss(other_student, other_teacher, images, patches, labels, 5, CONFIG,
                              augment=no_augmentation)
        actual.backward()
        torch.testing.assert_close(actual, expected)
        for a, b in zip(other_student.parameters(), student.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        for a, b in zip(other_teacher.parameters(), teacher.parameters()):
            torch.testing.assert_close(a, b)

    def test_rgb_bridge_matches_native_bgr_augmentation(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("upstream_transforms", ROOT / "references/crots/utils/transformsgpu.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        seed(40)
        rgb = torch.rand(2, 3, 32, 32)
        mean, std = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None], torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        bgr_mean = torch.tensor([104.00698793, 116.66876762, 122.67891434])
        bgr = rgb.flip(1)*255 - bgr_mean[None, :, None, None]
        seed(4)
        expected, _ = module.color_jitter(np.random.uniform(), bgr_mean, data=bgr)
        expected, _ = module.gaussian_blur(np.random.uniform(), data=expected)
        expected = (((expected + bgr_mean[None, :, None, None])/255).flip(1)-mean)/std
        seed(4)
        actual = augment_rgb((rgb-mean)/std)
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)

    def test_prepare_distill_adapt_and_completed_stage_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(2):
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / f"{i}.png")
                rows.append({"image": f"{i}.png", "label": ""})
            config = dict(CONFIG, image_size=32, batch_size=2, distill_epochs=1, adapt_epochs_per_round=1, rounds=1)
            args = SimpleNamespace(output=root / "run", source_checkpoint=root / "source.pth", data_root=root,
                                   backbone=root / "backbone.pth", seed=1, workers=0, smoke=True)
            args.output.mkdir()
            def model(_): return TinyEncoder(), TinyClassifier()
            with patch.object(runner, "make_model", side_effect=model), patch.object(runner, "initialize_backbone"):
                net = runner.Segmenter()
                torch.save({"task": "r2u", "stage": "source", "num_classes": 7,
                            "normalization": "imagenet_rgb_planes", "source_split": "loveda_rural_train",
                            "feature_extractor": net.encoder.state_dict(), "classifier": net.classifier.state_dict()}, args.source_checkpoint)
                protocol = {"signature": "test"}
                device = torch.device("cpu")
                runner.prepare(args, config, rows, device, protocol)
                runner.train_stage(args, config, rows, device, protocol, 0)
                runner.train_stage(args, config, rows, device, protocol, 1)
                last = args.output / "adapt_round1/last.pth"
                before = runner.sha256_file(last)
                runner.train_stage(args, config, rows, device, protocol, 1)
                self.assertEqual(before, runner.sha256_file(last))
                state = runner.load_checkpoint(last)
                self.assertEqual(state["completed_epochs"], 1)
                self.assertIsNotNone(state["teacher"])
                with self.assertRaisesRegex(ValueError, "different protocol"):
                    runner.load_student(last, device, "wrong")

    def test_epoch_resume_matches_uninterrupted_distillation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            pseudo = output / "pseudo"
            pseudo.mkdir(parents=True)
            rows = []
            for i in range(2):
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / f"{i}.png")
                Image.new("L", (32, 32), i).save(pseudo / f"{i:06d}.png")
                rows.append({"image": f"{i}.png", "label": ""})
            (pseudo / "complete.json").write_text(json.dumps({"protocol_sha256": "resume"}))
            config = dict(CONFIG, image_size=32, batch_size=2, distill_epochs=2)
            args = SimpleNamespace(output=output, data_root=root, backbone=root/"unused", seed=3, workers=0)
            snapshots = []
            original_save = runner.write_checkpoint
            def capture(path, payload):
                original_save(path, payload)
                if payload["completed_epochs"] == 1:
                    snapshots.append(path.read_bytes())
            with patch.object(runner, "make_model", side_effect=lambda _: (TinyEncoder(), TinyClassifier())), patch.object(runner, "initialize_backbone"):
                seed(10)
                with patch.object(runner, "write_checkpoint", side_effect=capture):
                    runner.train_stage(args, config, rows, torch.device("cpu"), {"signature": "resume"}, 0)
                path = output / "distill/last.pth"
                expected = runner.load_checkpoint(path)["student"]
                path.write_bytes(snapshots[0])
                runner.train_stage(args, config, rows, torch.device("cpu"), {"signature": "resume"}, 0)
                actual = runner.load_checkpoint(path)["student"]
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
