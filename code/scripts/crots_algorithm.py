"""CROTS SMHM loss port; see references/crots and REPRODUCE_CROTS_R2U.md.

Preserves the public code's raw-logit threshold and pre-optimizer EMA update.
The RGB normalization bridge retains the existing DGLE-protocol source model.
"""
import random

import kornia
import numpy as np
import torch
import torch.nn.functional as F


def spatial_mix(images, predictions, beta=0.25):
    if images.shape[0] < 2:
        raise ValueError("CROTS spatial mixing requires batch size >= 2")
    mixed_images, mixed_predictions = [], []
    height, width = images.shape[-2:]
    for i in range(images.shape[0]):
        others = random.choices([j for j in range(images.shape[0]) if j != i], k=3)
        ids = [i] + others
        x = int((0.2 + np.random.beta(beta, beta) * 0.6) * width) if beta > 0 else width // 2
        y = int((0.2 + np.random.beta(beta, beta) * 0.6) * height) if beta > 0 else height // 2
        def stitch(values):
            top = torch.cat((values[ids[0], :, :y, :x], values[ids[1], :, :y, x:]), -1)
            bottom = torch.cat((values[ids[2], :, y:, :x], values[ids[3], :, y:, x:]), -1)
            return torch.cat((top, bottom), -2)
        mixed_images.append(stitch(images))
        mixed_predictions.append(stitch(predictions))
    images, predictions = torch.stack(mixed_images), torch.stack(mixed_predictions)
    if np.random.uniform() < 0.5:
        images, predictions = images.flip(-1), predictions.flip(-1)
    return images, predictions


def augment_rgb(images):
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    # Upstream applies ColorJitter to BGR values. Bridge explicitly instead of
    # applying BGR mean-subtraction directly to our RGB-normalized tensors.
    if np.random.uniform() > 0.2:
        bgr = (images * std + mean).flip(1)
        bgr = kornia.augmentation.ColorJitter(0.25, 0.25, 0.25, 0.25)(bgr)
        images = (bgr.flip(1) - mean) / std
    if np.random.uniform() > 0.5:
        sigma = np.random.uniform(0.15, 1.15)
        kernels = tuple(int(np.floor(np.ceil(0.1 * n) - 0.5 + np.ceil(0.1 * n) % 2))
                        for n in images.shape[-2:])
        images = kornia.filters.GaussianBlur2d(kernels, (sigma, sigma))(images)
    return images


def smhm_loss(student, teacher, images, patch_images, patch_labels, step, config,
              augment=augment_rgb):
    with torch.no_grad():
        predictions = teacher(images, size=images.shape[-2:])
    mixed, logits = spatial_mix(images, predictions, config["position_beta"])
    confidence, labels = logits.max(1)
    augmented = augment(mixed)
    weight = (confidence >= config["confidence_threshold"]).float().mean()
    output = student(augmented, size=labels.shape[-2:])
    consistency = weight * F.cross_entropy(output, labels, reduction="none").mean()
    patch_output = student(patch_images, size=patch_labels.shape[-2:])
    hard = config["hard_patch_weight"] * F.cross_entropy(patch_output, patch_labels)
    # Upstream performs this before the optimizer step, including alpha=0 at step 0.
    alpha = min(1 - 1 / (step + 1), config["ema_decay"])
    with torch.no_grad():
        for ema, parameter in zip(teacher.parameters(), student.parameters()):
            ema.mul_(alpha).add_(parameter, alpha=1 - alpha)
    return consistency + hard, {"consistency": float(consistency.detach()),
                               "hard_patch": float(hard.detach()),
                               "unlabeled_weight": float(weight), "ema_alpha": alpha}
