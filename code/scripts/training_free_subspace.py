"""Closed-form frozen-feature scorers; no dataset, labels, or training imports."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


ROUTES = ('B0', 'B1', 'B2', 'M1', 'C1')


def tile_moments(features, tiles):
    """Population covariance: each represented tile has total weight 1/K."""
    features = features.double()
    unique, inverse, counts = torch.unique(tiles, return_inverse=True, return_counts=True)
    weights = 1. / (len(unique) * counts[inverse].double())
    mean = (features * weights[:, None]).sum(0)
    centered = features - mean
    covariance = centered.T @ (centered * weights[:, None])
    return mean, (covariance + covariance.T) / 2, float(1 / weights.square().sum())


def lowrank_score(features, mean, basis, amplitude, floor):
    centered = features - mean
    quadratic = centered.square().sum(-1) / floor
    projection = centered @ basis
    quadratic -= (projection.square() * (amplitude / (floor * (floor + amplitude)))).sum(-1)
    logdet = features.shape[-1] * math.log(floor) + torch.log1p(amplitude / floor).sum()
    return -.5 * (quadratic + logdet)


def fit_scorers(bank, rank=8, random_seed=1729, min_sites=256, min_tiles=3):
    means, covariances, classes, geometry = [], [], [], {}
    for cls, item in sorted(bank.items()):
        x, tiles = item['features'], item['tiles']
        support = {'sites': len(x), 'tiles': len(torch.unique(tiles))}
        support['supported'] = support['sites'] >= min_sites and support['tiles'] >= min_tiles
        geometry[str(cls)] = support
        if not support['supported']:
            continue
        mean, covariance, effective_n = tile_moments(x, tiles)
        classes.append(cls)
        means.append(mean)
        covariances.append(covariance)
        support['weight_effective_sample_size'] = effective_n
    if len(classes) < 2:
        return {'classes': classes, 'active': False, 'rank': rank}, geometry
    means, covariances = torch.stack(means), torch.stack(covariances)
    d = means.shape[1]
    floor = max(float(covariances.diagonal(dim1=-2, dim2=-1).sum(-1).mean() / d), 1e-6)
    bases, amplitudes, random_bases = [], [], []
    generator = torch.Generator().manual_seed(random_seed)
    for cls, covariance in zip(classes, covariances):
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.arange(d - 1, max(-1, d - rank - 1), -1)
        order = order[eigenvalues[order] > floor]
        basis, amplitude = eigenvectors[:, order], eigenvalues[order] - floor
        bases.append(basis.float())
        amplitudes.append(amplitude.float())
        random_bases.append(torch.linalg.qr(torch.randn(d, len(order), generator=generator,
                                                       dtype=torch.float64), mode='reduced')[0].float())
        spectrum = eigenvalues.clamp_min(0)
        fraction = spectrum / spectrum.sum().clamp_min(1e-30)
        info = geometry[str(cls)]
        info.update(retained_rank=len(order), spectrum=spectrum.flip(0).tolist(),
                    explained_variance=float(spectrum[order].sum() / spectrum.sum().clamp_min(1e-30)),
                    effective_rank=float((-(fraction * fraction.clamp_min(1e-30).log()).sum()).exp()))
        item = bank[cls]
        # Independent tile halves, with the same class-level tile weighting.
        unique = torch.unique(item['tiles'])
        halves = []
        for group in (unique[::2], unique[1::2]):
            mask = torch.isin(item['tiles'], group)
            if len(group) < 2 or int(mask.sum()) < rank + 1 or not len(order):
                halves = []
                break
            _, half_covariance, _ = tile_moments(item['features'][mask], item['tiles'][mask])
            _, vectors = torch.linalg.eigh(half_covariance)
            halves.append(vectors[:, -len(order):])
        if halves:
            cosines = torch.linalg.svdvals(halves[0].T @ halves[1]).clamp(0, 1)
            info['tile_half_principal_angles_degrees'] = torch.rad2deg(torch.acos(cosines)).tolist()
            info['tile_half_projection_overlap'] = float(cosines.square().mean())
        else:
            info['tile_half_projection_overlap'] = None
    shared = .9 * covariances.mean(0) + .1 * floor * torch.eye(d, dtype=torch.float64)
    # The shared z^T inv(Sigma) z and logdet cancel inside softmax.
    precision_means = torch.linalg.solve(shared, means.T).T
    model = {'active': True, 'classes': classes, 'means': means.float(), 'floor': floor,
             'bases': bases, 'amplitudes': amplitudes, 'random_bases': random_bases,
             'shared_weights': precision_means.float(),
             'shared_bias': (-.5 * (means * precision_means).sum(-1)).float(),
             'rank': rank, 'random_seed': random_seed}
    return model, geometry


def scores(features, model, route):
    if not model['active']:
        raise ValueError('At least two supported classes required')
    if route == 'B2':
        return features @ model['shared_weights'].T + model['shared_bias']
    output = []
    for index, mean in enumerate(model['means']):
        basis = model['random_bases' if route == 'C1' else 'bases'][index]
        amplitude = model['amplitudes'][index]
        if route == 'B1':
            basis, amplitude = basis[:, :0], amplitude[:0]
        elif route not in ('M1', 'C1'):
            raise ValueError(route)
        output.append(lowrank_score(features, mean, basis, amplitude, model['floor']))
    return torch.stack(output, -1)


def calibrate(features, model):
    if not model['active']:
        return {route: None for route in ROUTES[1:]}
    result = {}
    for route in ROUTES[1:]:
        values = scores(features, model, route).topk(2, dim=-1).values
        # NumPy median averages the two middle values for an even site count.
        gap = (values[:, 0] - values[:, 1]).double().cpu().numpy()
        if not np.isfinite(gap).all() or not len(gap):
            raise FloatingPointError('Invalid unlabeled calibration scores')
        result[route] = max(float(np.median(gap)), 1e-6)
    return result


def blend_log_probabilities(logits, geometry_scores, classes, temperature, alpha=.5):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    if alpha == 0 or len(classes) < 2:
        return logits  # Exact native parity, including ties and extreme logits.
    source = logits.log_softmax(-1)
    output = source.clone()
    selected = source[..., classes]
    mass = selected.logsumexp(-1, keepdim=True)
    candidate = (geometry_scores / temperature).log_softmax(-1) + mass
    if alpha == 1:
        output[..., classes] = candidate
    else:
        output[..., classes] = torch.logaddexp(selected + math.log1p(-alpha),
                                               candidate + math.log(alpha))
    return output


def predict_routes(fused, native_logits, model, temperatures, size, alpha=.5):
    flat_features = F.normalize(fused, dim=1).permute(0, 2, 3, 1).reshape(-1, fused.shape[1])
    flat_logits = native_logits.permute(0, 2, 3, 1).reshape(-1, native_logits.shape[1])
    source = F.interpolate(native_logits, size=size, mode='bilinear', align_corners=False)
    yield 'B0', source
    for route in ROUTES[1:]:
        if not model['active'] or alpha == 0:
            yield route, source
            continue
        mixed = blend_log_probabilities(flat_logits, scores(flat_features, model, route),
                                        model['classes'], temperatures[route], alpha)
        mixed = mixed.reshape(native_logits.shape[0], *native_logits.shape[-2:], -1).permute(0, 3, 1, 2)
        yield route, F.interpolate(mixed, size=size, mode='bilinear', align_corners=False)


def confusion(prediction, labels, classes=6):
    valid = labels != 255
    if np.any((labels[valid] < 0) | (labels[valid] >= classes)):
        raise ValueError('Mask must contain native IDs 0..5 or ignore255; do not remap twice')
    if np.any((prediction[valid] < 0) | (prediction[valid] >= classes)):
        raise ValueError('Invalid prediction class')
    return np.bincount(labels[valid] * classes + prediction[valid], minlength=classes ** 2).reshape(classes, classes)


def metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    tp, gt, predicted = matrix.diagonal(), matrix.sum(1), matrix.sum(0)
    def divide(a, b):
        return np.divide(a, b, out=np.full(len(tp), np.nan), where=b > 0)
    iou = divide(tp, gt + predicted - tp)
    def nullable(x):
        return [float(v) if np.isfinite(v) else None for v in x]
    def mean_present(x):
        return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else None
    return {'miou6': mean_present(iou), 'miou5': mean_present(iou[:5]),
            'iou': nullable(iou), 'precision': nullable(divide(tp, predicted)),
            'recall': nullable(divide(tp, gt)), 'f1': nullable(divide(2 * tp, gt + predicted)),
            'valid_pixels': int(matrix.sum()), 'confusion_matrix': matrix.tolist()}
