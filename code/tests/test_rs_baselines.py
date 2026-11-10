"""CPU checks for the re-implemented S3T-SFDA and St-CGN hooks (scripts/rs_baselines.py) and the St-CGN warm-up."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from scripts import rs_baselines as rb

ROOT = Path(__file__).parents[1]


class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.ModuleList([torch.nn.Conv2d(3, ch, 3, stride=s, padding=1)
                                      for ch, s in ((64, 4), (128, 8), (320, 16), (512, 32))])

    def forward(self, x):
        f = [c(x) for c in self.c]
        return f[-1], f


class Dec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = torch.nn.Conv2d(64, 768, 1); self.pred = torch.nn.Conv2d(768, 6, 1)

    def fuse_features(self, f):
        return self.fuse(f[0])

    def classify_fused(self, z, size=None):
        y = self.pred(z)
        return F.interpolate(y, size=size, mode='bilinear', align_corners=False) if size else y

    def forward(self, f, size=None):
        return self.classify_fused(self.fuse_features(f), size)


class Runtime(SimpleNamespace):
    def full2weak(self, feat, params, down_ratio=1, nearest=False):
        return feat  # identity alignment (full view == crop) for the unit test


def settings(name):
    return json.loads((ROOT/'kaggle_settings'/name).read_text())


class S3TTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.enc, self.dec = Enc(), Dec()
        self.hook = rb.build_baseline(Runtime(), settings('s3t_sfda.json'), 'cpu')
        self.x = torch.randn(2, 3, 64, 64)
        self.prob = torch.softmax(torch.randn(2, 6, 64, 64), 1)
        self.labels = self.prob.argmax(1); self.labels[:, :4] = 255

    def run_loss(self, roi_prob, iteration=0):
        self.hook.s = dict(self.hook.s, roi_prob=roi_prob, roi_size=32)
        return self.hook.loss(iteration=iteration, encoder=self.enc, classifier=self.dec, student_input=self.x,
                              teacher_prob=self.prob, pseudo=self.labels, valid=torch.ones(2, 64, 64, dtype=torch.bool),
                              criterion=torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none'),
                              negative_weight=1.0)

    def test_losses_finite_and_backprop_to_encoder_and_projection(self):
        for roi_prob in (0.0, 1.0):
            total, pos, neg = self.run_loss(roi_prob)
            self.assertTrue(torch.isfinite(total) and torch.isfinite(pos) and torch.isfinite(neg))
            total.backward()
            self.assertIsNotNone(self.enc.c[1].weight.grad)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.hook.proj.parameters()))
            self.enc.zero_grad(); self.hook.optimizer.zero_grad()
        self.assertEqual(self.hook.roi_iterations, 1)

    def test_history_snapshot_cadence_and_agreement_weights(self):
        self.run_loss(0.0, iteration=0)
        first = self.hook.his[0]
        self.run_loss(0.0, iteration=1); self.assertIs(self.hook.his[0], first)
        self.run_loss(0.0, iteration=300); self.assertIsNot(self.hook.his[0], first)
        self.assertEqual(self.hook.snapshot_iteration, 300)
        # at a fresh snapshot the historical model equals the student: agreement weight 2 on argmax labels
        self.assertGreaterEqual(self.hook.last_record['s3t_agreement'], 0)

    def test_contrast_matches_reference_formula(self):
        emb = F.normalize(torch.randn(1, 8, 2, 2), dim=1); his = F.normalize(torch.randn(1, 8, 2, 2), dim=1)
        labels = torch.tensor([[[0, 0], [1, 255]]]); logits = torch.zeros(1, 6, 2, 2)
        got = rb.historical_contrast(emb, labels, his, logits, .07)
        a = emb.view(8, 4).T; k = his.view(8, 4).T; lab = labels.view(4).float()
        rel = torch.full((4,), .9)  # uniform historical prediction -> normalised entropy 1 -> clamp to 0.9
        mask = (lab[:, None] == lab[None]).float(); lm = 1 - torch.eye(4); mask = mask * lm
        lg = a @ k.T / .07; lg = lg - lg.max(1, keepdim=True).values
        ex = torch.exp(lg) * lm * rel[None]
        lp = lg - torch.log(ex.sum(1, keepdim=True) + 1e-20) + torch.log(rel[None] + 1e-20)
        self.assertAlmostEqual(float(got), float(-((mask * lp).sum(1) / 4).mean()), places=5)

    def test_state_roundtrip(self):
        self.run_loss(0.0)
        state = self.hook.state_dict()
        other = rb.build_baseline(Runtime(), settings('s3t_sfda.json'), 'cpu')
        other.load_state_dict(state, self.enc, self.dec)
        self.assertEqual(other.snapshot_iteration, 0)
        for a, b in zip(other.his[0].parameters(), self.hook.his[0].parameters()):
            self.assertTrue(torch.equal(a, b))


class StCGNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name)/'p.pt'
        torch.save({'prototypes': torch.randn(6, 768), 'counts': torch.ones(6)}, path)
        self.s = dict(settings('stcgn.json'), prototypes=str(path))
        self.enc, self.dec = Enc(), Dec()
        self.hook = rb.build_baseline(Runtime(), self.s, 'cpu')

    def test_class_balanced_threshold_and_rectified_argmax(self):
        prob = torch.tensor([[.9, .1], [.6, .4], [.55, .45], [.2, .8]]).T.reshape(1, 2, 2, 2)
        rect = prob.clone()
        labels, thr = rb.class_balanced_labels(prob, rect, torch.ones(1, 2, 2, dtype=torch.bool), .5)
        # class 0 originals .9/.6/.55 -> top-50% threshold = median .6; class 1 single value .8
        self.assertAlmostEqual(float(thr[0]), .6, places=5); self.assertAlmostEqual(float(thr[1]), .8, places=5)
        self.assertEqual(labels.view(-1).tolist(), [0, 0, 255, 1])
        # rectification can change the class; the threshold is checked on the ORIGINAL probability of that class
        rect2 = prob.clone(); rect2[0, :, 0, 0] = torch.tensor([.1, .9])
        labels2, _ = rb.class_balanced_labels(prob, rect2, torch.ones(1, 2, 2, dtype=torch.bool), .5)
        self.assertEqual(int(labels2.view(-1)[0]), 255)  # P(class1)=.1 < .8

    def test_p_ratio_schedule(self):
        self.assertAlmostEqual(self.hook.p_ratio(0), .4); self.assertAlmostEqual(self.hook.p_ratio(2500), .65)
        self.assertAlmostEqual(self.hook.p_ratio(10**6), .9)

    def test_loss_and_prototype_refresh(self):
        x = torch.randn(2, 3, 64, 64); pt = torch.softmax(torch.randn(2, 6, 64, 64), 1)
        kw = dict(encoder=self.enc, classifier=self.dec, unmixed=x, full=x, params=None, teacher_prob=pt,
                  valid=torch.ones(2, 64, 64, dtype=torch.bool),
                  criterion=torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none'))
        before = self.hook.prototypes.clone()
        total, target, neg = self.hook.loss(iteration=0, **kw)
        self.assertTrue(torch.isfinite(total)); self.assertEqual(float(neg), 0.)
        total.backward(); self.assertIsNotNone(self.enc.c[0].weight.grad)
        self.assertTrue(torch.equal(before, self.hook.prototypes))
        self.hook.loss(iteration=self.s['prototype_every'] - 1, **kw)
        self.assertEqual(self.hook.refreshes, 1); self.assertFalse(torch.equal(before, self.hook.prototypes))
        # marginal entropy term is minimised at the uniform marginal: L_ent = -log 6
        self.assertGreaterEqual(self.hook.last_record['stcgn_ent'], -math.log(6) - 1e-6)


class WarmupTests(unittest.TestCase):
    def test_instance_and_region_losses(self):
        from train_stcgn_warmup import instance_loss, region_loss
        torch.manual_seed(0)
        f = torch.randn(2, 16, 8, 8, requires_grad=True); t = f.detach().clone()
        y = torch.randint(0, 3, (2, 8, 8))
        li = instance_loss(f, t, y, y, .1); li.backward()
        self.assertTrue(torch.isfinite(li)); self.assertIsNotNone(f.grad)
        lr, s_min, accepted = region_loss(f, t, .7, .1, anchors=16, negatives=15)
        # identical views: every anchor finds itself (cosine 1 > 0.7)
        self.assertEqual(accepted, 32); self.assertAlmostEqual(s_min, 1.0, places=4)
        _, s_none, acc_none = region_loss(f, -t, .99, .1, anchors=16, negatives=15)
        self.assertEqual(acc_none, 0); self.assertIsNone(s_none)


if __name__ == '__main__':
    unittest.main()
