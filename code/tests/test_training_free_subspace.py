"""Run: PYTHONPATH=. python tests/test_training_free_subspace.py (NumPy/PyTorch)."""
import math
import json

import numpy as np
import torch
from torch.nn import functional as F

from scripts.training_free_subspace import (blend_log_probabilities, confusion, fit_scorers,
    lowrank_score, metrics, scores, tile_moments)


def check():
    torch.manual_seed(37)
    d, floor = 7, .12
    mean, features = torch.randn(d, dtype=torch.float64), torch.randn(11, d, dtype=torch.float64)
    basis = torch.linalg.qr(torch.randn(d, 3, dtype=torch.float64)).Q
    amplitude = torch.tensor([.8, .3, .1], dtype=torch.float64)
    covariance = floor * torch.eye(d, dtype=torch.float64) + basis @ amplitude.diag() @ basis.T
    centered = features - mean
    dense = -.5 * ((centered @ torch.linalg.inv(covariance) * centered).sum(-1) + torch.logdet(covariance))
    assert torch.allclose(lowrank_score(features, mean, basis, amplitude, floor), dense, atol=1e-10)
    isotropic = -.5 * (centered.square().sum(-1) / floor + d * math.log(floor))
    assert torch.allclose(lowrank_score(features, mean, basis[:, :0], amplitude[:0], floor), isotropic)
    x = torch.tensor([[0., 0.], [2., 0.], [10., 0.]])
    mu, cov, _ = tile_moments(x, torch.tensor([0, 0, 1]))
    assert torch.allclose(mu, torch.tensor([5.5, 0.], dtype=torch.float64))
    assert torch.allclose(cov, torch.tensor([[20.75, 0.], [0., 0.]], dtype=torch.float64))
    logits = torch.randn(17, 6)
    logits[0] = torch.tensor([-1000., 1000., -2000., 1000., 0., 1.])
    classes = [0, 2, 4]
    adjusted = blend_log_probabilities(logits, torch.randn(17, 3), classes, .5)
    p0, p1 = logits.softmax(-1), adjusted.softmax(-1)
    assert torch.isfinite(adjusted).all()
    assert torch.allclose(p1.sum(-1), torch.ones(17), atol=1e-6)
    assert torch.allclose(p0[:, [1, 3, 5]], p1[:, [1, 3, 5]], atol=1e-6)
    assert torch.equal(blend_log_probabilities(logits, None, classes, 1., alpha=0), logits)
    assert torch.equal(blend_log_probabilities(logits, None, [0], 1.), logits)
    bank = {c: {'features': F.normalize(torch.randn(24, d), dim=-1),
                'tiles': torch.arange(24) % 4} for c in range(3)}
    zero, _ = fit_scorers(bank, rank=0, min_sites=1)
    assert torch.allclose(scores(features.float(), zero, 'M1'), scores(features.float(), zero, 'B1'))
    fitted, _ = fit_scorers(bank, rank=3, min_sites=1)
    covariances = torch.stack([tile_moments(v['features'], v['tiles'])[1] for v in bank.values()])
    means = torch.stack([tile_moments(v['features'], v['tiles'])[0] for v in bank.values()])
    shared = .9 * covariances.mean(0) + .1 * fitted['floor'] * torch.eye(d, dtype=torch.float64)
    dense_shared = torch.stack([-.5 * (((features - m) @ torch.linalg.inv(shared) *
                                       (features - m)).sum(-1) + torch.logdet(shared)) for m in means], -1)
    linear = scores(features.float(), fitted, 'B2')
    assert torch.allclose(linear - linear[:, :1], (dense_shared - dense_shared[:, :1]).float(), atol=5e-5)
    for random_basis in fitted['random_bases']:
        assert torch.allclose(random_basis.T @ random_basis, torch.eye(random_basis.shape[1]), atol=1e-6)
    labels, pred = np.array([[0, 1, 255], [5, 0, 1]]), np.array([[0, 5, 2], [0, 0, 1]])
    cm = confusion(pred, labels)
    manual = np.zeros((6, 6), dtype=np.int64)
    for truth, guess in zip(labels.flatten(), pred.flatten()):
        if truth != 255:
            manual[truth, guess] += 1
    assert np.array_equal(cm, manual) and metrics(cm)['valid_pixels'] == 5
    assert metrics(cm)['iou'][5] == 0 and cm[1, 5] == 1
    empty = metrics(np.zeros((6, 6), dtype=np.int64))
    assert empty['valid_pixels'] == 0 and empty['miou6'] is None and empty['miou5'] is None
    json.dumps(empty, allow_nan=False)
    try:
        confusion(pred, np.array([[0, 1, 7], [5, 0, 1]]))
        raise AssertionError('Invalid label accepted')
    except ValueError:
        pass
    weights = torch.randn(4, d, dtype=torch.float64)
    projection = weights.T @ torch.linalg.pinv(weights @ weights.T) @ weights
    assert torch.allclose(features @ weights.T, features @ projection @ weights.T, atol=1e-10)
    print('PASS: dense/low-rank/shared equivalence; tile weighting; rank0; alpha0; mass; masks; empty CM; row-space')


if __name__ == '__main__':
    check()
