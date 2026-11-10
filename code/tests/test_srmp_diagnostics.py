"""CPU tests. Synthetic references are fixtures, never a production fallback."""
import copy
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from core.configs import cfg
import srmp_diagnostics as d
import train_SRMP as m


class DiagnosticMathTests(unittest.TestCase):
    def test_histogram_orientation_ties_and_empty(self):
        perfect = d.histogram_metrics([2, 0, 2], [0, 0, 2])
        self.assertEqual(perfect["error_detection_auroc_binned"], 1)
        self.assertEqual(perfect["error_detection_ap_binned"], 1)
        self.assertEqual(perfect["risk_at_retained_fraction"]["0.5"], 0)
        self.assertEqual(d.histogram_metrics([4], [2])["error_detection_auroc_binned"], 0.5)
        self.assertIsNone(d.histogram_metrics([2], [0])["error_detection_auroc_binned"])
        self.assertIsNone(d.histogram_metrics([0], [0])["error_rate"])

    def test_agreement_is_not_correctness(self):
        teacher = torch.tensor([[[[0.9, 0.9, 0.9, 0.9]], [[0.1, 0.1, 0.1, 0.1]]]])
        masked = torch.tensor([[[[0.9, 0.9, 0.1, 0.1]], [[0.1, 0.1, 0.9, 0.9]]]])
        truth = torch.tensor([[[0, 1, 0, 1]]])
        probe = d.ErrorProbe(2, bins=16)
        probe.update("image", truth, teacher, masked, teacher, torch.ones_like(truth, dtype=torch.bool),
                     m.relational_cosine_map(masked, teacher, torch.eye(2)))
        result = probe.summary()
        self.assertEqual(result["agreement"]["teacher_masked"]["table"], [[1, 1], [1, 1]])
        self.assertEqual(result["agreement"]["teacher_masked"]["p_error_given_agreement"], 0.5)
        self.assertEqual(result["agreement"]["teacher_masked"]["consistent_wrong_fraction_of_errors"], 0.5)
        self.assertEqual(probe.image_rows[0]["high_conf_agree_wrong"], 1)
        json.dumps(result, allow_nan=False)

    def test_ignore_and_unobserved_cells_are_not_counted(self):
        q = torch.full((1, 2, 1, 3), 0.5)
        probe = d.ErrorProbe(2)
        probe.update("empty", torch.full((1, 1, 3), 255), q, q, q,
                     torch.zeros(1, 1, 3, dtype=torch.bool), torch.zeros(1, 1, 3))
        report = probe.summary()
        self.assertIsNone(report["agreement"]["teacher_masked"]["p_error_given_agreement"])
        json.dumps(report, allow_nan=False)

    def test_region_discovery_and_neighbours(self):
        features = torch.zeros(4, 6, 2)
        features[:, :3, 0], features[:, 3:, 1] = 1, 1
        regions, embeddings = d.discover_regions(features, clusters=2, min_size=2)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(torch.unique(regions[:, :3])), 1)
        self.assertFalse(torch.equal(regions[:, :3], regions[:, 3:]))
        image_ids = torch.tensor([0, 0, 1, 2])
        vectors = torch.tensor([[1., 0], [1., 0], [0.99, 0.01], [-1., 0]])
        graph = d.neighbour_graph(vectors, image_ids, bank_size=4, k=5, seed=1, device="cpu")
        for row, ids in enumerate(graph["index"]):
            self.assertTrue((image_ids[ids[ids >= 0]] != image_ids[row]).all())
        self.assertTrue((image_ids[graph["random_index"]] != image_ids).all())
        one_image = d.neighbour_graph(vectors, torch.zeros(4, dtype=torch.long), 4, 2, 1, "cpu")
        self.assertTrue((one_image["index"] == -1).all())

    def test_collapsed_teacher_is_visible_in_object_metrics(self):
        stats = {"probability": torch.tensor([[1., 0], [1., 0]]), "label": torch.zeros(2, dtype=torch.long),
                 "purity": torch.ones(2), "patches": torch.ones(2), "gt_valid": torch.ones(2),
                 "gt_purity": torch.ones(2), "gt_label": torch.tensor([0, 1]), "accuracy": torch.tensor([1., 0])}
        graph = {"index": torch.tensor([[1], [0]]), "similarity": torch.ones(2, 1),
                 "random_index": torch.tensor([1, 0]), "bank_size": 2}
        result = d.object_summary(stats, graph, 0.85, 0.8)
        self.assertEqual(result["teacher_cross_image_agreement"], 1)
        self.assertEqual(result["random_cross_image_agreement"], 1)
        self.assertEqual(result["teacher_class_entropy"], 0)
        self.assertEqual(result["reference_pair_gt_same_class_rate"], 0)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8), nn.Dropout2d(0.2))

    def forward(self, x):
        f = self.net(x)
        return f, f


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_list = nn.ModuleList([nn.Conv2d(8, 19, 1)])
        nn.init.zeros_(self.conv2d_list[0].weight)
        nn.init.zeros_(self.conv2d_list[0].bias)

    def forward(self, f):
        return self.conv2d_list[0](f)


class SyntheticReference:
    calls = 0

    def __init__(self, settings, device):
        self.metadata = {"model": "synthetic-test-fixture", "test_only": True}

    def features(self, image):
        SyntheticReference.calls += 1
        features = torch.zeros(4, 6, 4)
        features[:, :3, 0], features[:, 3:, 1] = 1, 1
        return features


class AuditIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config = cfg.clone()
        self.config.defrost()
        m.add_srmp_config(self.config)
        self.config.OUTPUT_DIR = str(root)
        self.config.SRMP.WARMUP_ITERS = 1
        self.config.SRMP.BLOCK_SIZE = 4
        self.config.SRMP_DIAG.ENABLED = True
        self.config.SRMP_DIAG.PERIOD = 2
        self.config.SRMP_DIAG.INPUT_SIZE = (48, 32)
        self.config.SRMP_DIAG.REGION_CLUSTERS = 2
        self.config.SRMP_DIAG.DINO_MODEL = "synthetic-test-fixture"
        self.records = []
        for i in range(3):
            image, label = root / (str(i) + ".png"), root / (str(i) + "_gt.png")
            Image.fromarray(np.full((32, 48, 3), 50 + i, dtype=np.uint8)).save(image)
            gt = np.full((32, 48), 7, dtype=np.uint8)
            gt[:, 24:] = 8
            Image.fromarray(gt).save(label)
            self.records.append({"name": str(i), "image": str(image), "label": str(label)})
        self.encoder = Encoder().train()
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.classifier = Head().requires_grad_(False).eval()
        self.rebuilder = m.SpatialRebuilder(8, 16, 2).train()
        self.audit = d.TrainingDiagnostics(self.config, torch.device("cpu"), self.records, SyntheticReference)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_sweep_cache_temporal_metrics_and_no_training_effects(self):
        SyntheticReference.calls = 0
        modules = [self.encoder, self.classifier, self.teacher, self.rebuilder]
        states = [copy.deepcopy(module.state_dict()) for module in modules]
        modes = [child.training for module in modules for child in module.modules()]
        self.encoder.net[0].weight.grad = torch.ones_like(self.encoder.net[0].weight)
        rng_torch, rng_python, rng_numpy = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        self.audit.run(0, *modules, torch.eye(19), force=True)
        self.assertTrue(torch.equal(rng_torch, torch.get_rng_state()))
        self.assertEqual(rng_python, random.getstate())
        self.assertTrue(np.array_equal(rng_numpy[1], np.random.get_state()[1]))
        self.assertEqual(modes, [child.training for module in modules for child in module.modules()])
        for module, state in zip(modules, states):
            self.assertTrue(all(torch.equal(v, state[key]) for key, v in module.state_dict().items()))
        self.assertTrue((self.encoder.net[0].weight.grad == 1).all())
        self.assertEqual(SyntheticReference.calls, 3)
        initial = json.loads((self.audit.output / "audit_iter000000.json").read_text())
        self.assertEqual(initial["images"], 3)
        self.assertFalse(initial["rebuilder_has_passed_warmup"])
        self.assertEqual(initial["objects"]["regions"], 6)
        self.assertEqual(initial["objects"]["reference_pair_gt_same_class_rate"], 1)
        self.assertGreater(initial["direction1"]["agreement"]["teacher_masked"]["p_error_given_agreement"], 0)
        self.audit.run(1, *modules, torch.eye(19))
        self.assertFalse((self.audit.output / "audit_iter000001.json").exists())
        with torch.no_grad():
            self.classifier.conv2d_list[0].bias[1] = 10
        self.audit.run(2, *modules, torch.eye(19))
        self.assertEqual(SyntheticReference.calls, 3)
        later = json.loads((self.audit.output / "audit_iter000002.json").read_text())
        self.assertEqual(later["objects"]["temporal_label_flip_rate"], 1)
        self.assertEqual(later["objects"]["previous_iteration"], 0)
        self.assertTrue((self.audit.output / "pairs_iter000002.csv").is_file())
        # A new manager (resume) reuses the bank and finds a strictly earlier snapshot.
        resumed = d.TrainingDiagnostics(self.config, torch.device("cpu"), self.records, SyntheticReference)
        resumed.run(4, *modules, torch.eye(19))
        self.assertEqual(SyntheticReference.calls, 3)
        self.assertEqual(json.loads((resumed.output / "audit_iter000004.json").read_text())["objects"]["temporal_label_flip_rate"], 0)

    def test_ground_truth_never_constructs_the_reference(self):
        modules = [self.encoder, self.classifier, self.teacher, self.rebuilder]
        self.audit.run(0, *modules, torch.eye(19), force=True)
        graph = copy.deepcopy(self.audit.graph)
        fingerprint = self.audit.fingerprint
        for record in self.records:
            Image.fromarray(np.full((32, 48), 255, dtype=np.uint8)).save(record["label"])
        other = d.TrainingDiagnostics(self.config, torch.device("cpu"), self.records, SyntheticReference)
        other.run(2, *modules, torch.eye(19))
        self.assertEqual(fingerprint, other.fingerprint)
        self.assertTrue(torch.equal(graph["index"], other.graph["index"]))
        result = json.loads((other.output / "audit_iter000002.json").read_text())
        self.assertIsNone(result["direction1"]["agreement"]["teacher_masked"]["p_error_given_agreement"])

    def test_restore_modes_and_rng_on_failure(self):
        before = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with d.preserve_training_state(self.encoder):
                torch.rand(17)
                self.assertFalse(self.encoder.training)
                raise RuntimeError("test")
        self.assertTrue(self.encoder.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Optional DINO adapter dependencies not installed")
class DinoAdapterTests(unittest.TestCase):
    def test_real_dinov3_api_with_tiny_local_test_weights(self):
        # Random tiny weights test the real library/API only; no scientific
        # metric is claimed and production never constructs models this way.
        from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast

        with tempfile.TemporaryDirectory() as root:
            model = DINOv3ViTModel(DINOv3ViTConfig(
                hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=4, num_register_tokens=4, image_size=32, patch_size=16))
            model.save_pretrained(root)
            DINOv3ViTImageProcessorFast().save_pretrained(root)
            config = cfg.clone()
            m.add_srmp_config(config)
            config.SRMP_DIAG.DINO_MODEL = root
            config.SRMP_DIAG.LOCAL_FILES_ONLY = True
            config.SRMP_DIAG.DINO_LONG_SIDE = 64
            reference = d.DinoReference(config.SRMP_DIAG, torch.device("cpu"))
            result = reference.features(Image.fromarray(np.full((28, 56, 3), 100, dtype=np.uint8)))
            self.assertEqual(tuple(result.shape), (2, 4, 32))
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(torch.allclose(result.norm(dim=-1), torch.ones(2, 4), atol=1e-5))
            self.assertFalse(reference.model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in reference.model.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
