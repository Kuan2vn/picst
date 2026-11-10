"""Opt-in donor geometry and a target-only masked-context control."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def validate_geometry(config):
    if config.get('name') != 'dtst_geometry_20261016_v1' or config.get('donor_mode') not in ('baseline', 'independent', 'coupled'):
        raise ValueError('Unknown geometry experiment')
    if config.get('target_gsd_m') != .05:
        raise ValueError('Geometry scouts require the registered5cm target data')
    context = config.get('context')
    if context is not None:
        if (config['donor_mode'] != 'baseline' or context.get('kind') != 'mic_style' or
                not math.isfinite(context.get('weight', float('nan'))) or not 0 < context['weight'] <= 1 or
                not 0 < context.get('mask_ratio', 0) < 1 or type(context.get('block_size')) is not int or
                not 1 <= context['block_size'] <= 256 or not 0 < context.get('confidence_threshold', 0) < 1):
            raise ValueError('Invalid masked-context control')
    return config


def donor_scale(mode, recipient_scale, draw):
    if mode == 'baseline':
        return 1 + 1.5*draw
    if mode == 'independent':
        return 1 + draw
    if mode != 'coupled' or recipient_scale is None or not 1 <= recipient_scale <= 2:
        raise ValueError('Coupled donor needs the actual recipient scale')
    return recipient_scale


def empty_geometry_stats():
    return {key: np.zeros(6, dtype=np.float64) for key in
            ('proposed', 'accepted', 'skipped_size', 'paste_pixels', 'pitch_ratio_sum', 'resize_sum')}


def masked_context_loss(encoder, classifier, images, teacher_probability, valid, config):
    """No teacher/model state or extra optimizer; global RNG is in the full resume."""
    n, _, h, w = images.shape
    block = config['block_size']
    gh, gw = math.ceil(h/block), math.ceil(w/block)
    if gh*gw < 2:
        raise ValueError('Masked context requires at least two mask blocks')
    mask = torch.rand(n, 1, gh, gw, device=images.device) < config['mask_ratio']
    flat = mask.flatten(1)
    flat[:, 0] = True
    flat[:, -1] = False
    mask = F.interpolate(mask.float(), size=(h, w), mode='nearest').bool()
    masked = images.masked_fill(mask, 0)  # Normalized zero is the training-channel mean.
    probability = teacher_probability.detach()
    confidence, labels = probability.max(1)
    labels = labels.masked_fill(~valid.bool(), 255)
    quality = ((confidence >= config['confidence_threshold']) & valid.bool()).flatten(1).sum(1)
    quality = quality / valid.flatten(1).sum(1).clamp_min(1)
    logits = classifier(encoder(masked)[1], (h, w))
    loss_map = F.cross_entropy(logits, labels, ignore_index=255, reduction='none')
    loss = config['weight'] * (loss_map*quality[:, None, None]).sum()/valid.sum().clamp_min(1)
    record = dict(context_loss=float(loss.detach()), context_quality=float(quality.mean()),
                  context_mask_fraction=float(mask.float().mean()), context_weight=config['weight'],
                  context_teacher_images=n, context_student_images=n,
                  context_target_labels_used=False)
    return loss, record
