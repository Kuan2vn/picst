import math
import unittest

import torch
from scripts.gsd_reliability import (
    class_topq, normalized_js, selective_outer_loss, warp_to_native)


class GeometryTests(unittest.TestCase):
    def geometry(self, native=32, view=32, origin=0):
        return [{"native_size": native, "native_x": 0, "native_y": 0,
                 "view_size": view, "view_x": origin, "view_y": origin}]

    def test_identity(self):
        probability = torch.randn(1, 6, 32, 32).softmax(1)
        aligned, valid = warp_to_native(probability, self.geometry())
        self.assertTrue(valid.all())
        torch.testing.assert_close(aligned, probability, atol=1e-6, rtol=1e-5)
        self.assertLess(float(normalized_js(probability, aligned).max()), 1e-6)

    def test_wide_crop_coordinate_ramp(self):
        # View centers encode a known ground-coordinate ramp, independently of warp.
        axis = (torch.arange(32) + .5) * 58 / 32 - .5 - 13
        first = ((axis + 30) / 100)[None, None, None].expand(1, 1, 32, 32)
        probability = torch.cat((first, 1 - first), 1)
        aligned, valid = warp_to_native(probability, self.geometry(view=58, origin=-13))
        expected = ((torch.arange(32) + 30) / 100)[None, None].expand(1, 32, 32)
        self.assertTrue(valid.all())
        torch.testing.assert_close(aligned[:, 0], expected, atol=1e-6, rtol=1e-5)

    def test_small_crop_has_partial_support(self):
        probability = torch.ones(1, 2, 32, 32) / 2
        _, valid = warp_to_native(probability, self.geometry(view=18, origin=7))
        self.assertEqual(int(valid.sum()), 18 * 18)
        self.assertFalse(valid[0, 0, 0])
        self.assertTrue(valid[0, 16, 16])

    def test_invalid_observation_is_excluded(self):
        probability = torch.ones(1, 2, 32, 32) / 2
        support = torch.ones(1, 32, 32, dtype=torch.bool)
        support[:, :4] = False
        _, valid = warp_to_native(probability, self.geometry(), support)
        self.assertEqual(int(valid.sum()), 28 * 32)

    def test_js_bounds(self):
        p = torch.tensor([[[[1.]], [[0.]]]])
        q = 1 - p
        self.assertAlmostEqual(float(normalized_js(p, q)), 1, places=5)
        self.assertEqual(float(normalized_js(p, p)), 0)

    def test_identity_survives_weak_resize_crop_flip(self):
        logits = torch.randn(1, 6, 32, 32)
        aligned, _ = warp_to_native(logits, self.geometry(), normalize=False)
        def weak(value):
            return torch.nn.functional.interpolate(value, (49, 49), mode="bilinear",
                                                    align_corners=False)[:, :, 7:39, 9:41].flip(-1)
        native_probability = weak(logits).softmax(1)
        gsd_probability = weak(aligned).softmax(1)
        torch.testing.assert_close(native_probability, gsd_probability, atol=1e-6, rtol=1e-5)
        self.assertLess(float(normalized_js(native_probability, gsd_probability).max()), 1e-6)


class SelectorTests(unittest.TestCase):
    def test_per_class_budget_and_ties(self):
        labels = torch.tensor([[[0, 0, 0, 1, 1, 255]]])
        score = torch.ones_like(labels, dtype=torch.float)
        selected = class_topq(score, labels, torch.ones_like(labels, dtype=torch.bool), .5)
        self.assertEqual(selected.tolist(), [[[True, True, False, True, False, False]]])

    def test_q1_matches_baseline_loss_and_gradient(self):
        logits = torch.randn(1, 6, 4, 4, requires_grad=True)
        labels = torch.randint(0, 6, (1, 4, 4))
        labels[0, 0, 0] = 255
        donor = torch.full_like(labels, 255)
        donor[:, 1] = labels[:, 1]
        ce = torch.nn.functional.cross_entropy(logits, labels, ignore_index=255, reduction="none")
        weights = torch.rand_like(ce) + .5
        baseline = (ce * weights).mean()
        loss, _, _ = selective_outer_loss(
            ce, weights, labels, donor, labels != 255, weights, 1)
        self.assertTrue(torch.equal(loss, baseline))
        expected = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
        actual = torch.autograd.grad(loss, logits)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_donor_and_outside_overlap_keep_coefficients(self):
        loss_map = torch.ones(1, 1, 4, requires_grad=True)
        labels = torch.tensor([[[0, 0, 1, 1]]])
        pasted = torch.tensor([[[255, 255, 1, 255]]])
        eligible = torch.tensor([[[True, True, True, False]]])
        score = torch.tensor([[[.9, .1, .5, .4]]])
        loss, selected, _ = selective_outer_loss(
            loss_map, torch.ones_like(loss_map), labels, pasted, eligible, score, .5)
        self.assertEqual(selected.tolist(), [[[True, False, False, False]]])
        gradient = torch.autograd.grad(loss, loss_map)[0]
        torch.testing.assert_close(gradient, torch.tensor([[[.5, 0, .25, .25]]]))

    def test_empty_selection_is_finite_and_connected(self):
        loss_map = torch.zeros(1, 2, 2, requires_grad=True)
        labels = torch.full((1, 2, 2), 255)
        loss, _, _ = selective_outer_loss(
            loss_map, torch.ones_like(loss_map), labels, labels,
            labels != 255, loss_map.detach(), .5)
        self.assertTrue(math.isfinite(float(loss.detach())))
        loss.backward()
        self.assertIsNotNone(loss_map.grad)


if __name__ == "__main__":
    unittest.main()
