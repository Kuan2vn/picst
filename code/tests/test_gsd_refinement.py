import unittest
import torch
from scripts.gsd_reliability import selective_outer_loss, soft_js_outer_loss


class RefinementTests(unittest.TestCase):
    def inputs(self):
        loss = torch.ones(1, 1, 4, requires_grad=True)
        labels = torch.tensor([[[0, 0, 1, 1]]])
        pasted = torch.tensor([[[255, 255, 1, 255]]])
        support = torch.tensor([[[True, True, True, False]]])
        score = torch.tensor([[[.9, .1, .5, .4]]])
        return loss, labels, pasted, support, score

    def test_blend_preserves_unselected_and_donor_gradients(self):
        loss, labels, pasted, support, score = self.inputs()
        value, _, _ = selective_outer_loss(loss, torch.ones_like(loss), labels, pasted, support, score, .5, .25)
        gradient = torch.autograd.grad(value, loss)[0]
        torch.testing.assert_close(gradient, torch.tensor([[[1.25, .75, 1., 1.]]])/4)

    def test_zero_intervention_is_exact_raw_gradient(self):
        loss, labels, pasted, support, score = self.inputs()
        value, _, _ = selective_outer_loss(loss, torch.ones_like(loss), labels, pasted, support, score, .5, 0.)
        torch.testing.assert_close(torch.autograd.grad(value, loss)[0], torch.ones_like(loss)/4, atol=0, rtol=0)

    def test_soft_js_never_amplifies_and_does_not_backpropagate_teacher(self):
        loss, labels, pasted, support, _ = self.inputs()
        js = torch.tensor([[[0., 1., 1., 1.]]], requires_grad=True)
        value, eligible, coefficient = soft_js_outer_loss(loss, torch.ones_like(loss), labels, pasted, support, js)
        self.assertTrue((coefficient >= .5).all() and (coefficient <= 1).all())
        self.assertEqual(coefficient[0,0,0], 1.)
        self.assertEqual(coefficient[0,0,2], 1.)
        self.assertEqual(coefficient[0,0,3], 1.)
        value.backward()
        self.assertIsNone(js.grad)
        torch.testing.assert_close(loss.grad, coefficient/4)

    def test_invalid_intervention_is_rejected(self):
        loss, labels, pasted, support, score = self.inputs()
        with self.assertRaises(ValueError):
            selective_outer_loss(loss, torch.ones_like(loss), labels, pasted, support, score, .5, 1.1)


if __name__ == '__main__':
    unittest.main()
