"""Physical-view alignment and an outer-only selector for the existing SND loop."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def warp_to_native(probabilities, geometry, view_valid=None, normalize=True):
    """Map view pixel centers to the native crop; cropping is not invertible."""
    if len(geometry) != probabilities.shape[0]:
        raise ValueError("One geometry record is required per prediction")
    outputs, masks = [], []
    height, width = probabilities.shape[-2:]
    for index, record in enumerate(geometry):
        size = int(record["native_size"])
        crop = int(record["view_size"])
        if size < 1 or crop < 1:
            raise ValueError("Invalid crop size")
        axis = torch.arange(size, device=probabilities.device, dtype=torch.float32)
        x = (axis + record["native_x"] - record["view_x"] + .5) * width / crop - .5
        y = (axis + record["native_y"] - record["view_y"] + .5) * height / crop - .5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((2 * (xx + .5) / width - 1,
                            2 * (yy + .5) / height - 1), -1)[None]
        value = F.grid_sample(probabilities[index:index + 1].float(), grid,
                              mode="bilinear", padding_mode="zeros", align_corners=False)
        valid = ((xx >= 0) & (xx <= width - 1) & (yy >= 0) & (yy <= height - 1))
        if view_valid is not None:
            support = F.grid_sample(view_valid[index:index + 1, None].float(), grid,
                                    mode="bilinear", padding_mode="zeros", align_corners=False)
            valid &= support[0, 0] >= 1 - 1e-5
        if normalize:
            value /= value.sum(1, keepdim=True).clamp_min(1e-8)
        outputs.append(value)
        masks.append(valid)
    return torch.cat(outputs), torch.stack(masks)


def normalized_js(first, second):
    first = first.detach().float().clamp_min(1e-8)
    second = second.detach().float().clamp_min(1e-8)
    first = first / first.sum(1, keepdim=True)
    second = second / second.sum(1, keepdim=True)
    mean = .5 * (first + second)
    divergence = .5 * ((first * (first.log() - mean.log())).sum(1)
                       + (second * (second.log() - mean.log())).sum(1))
    return (divergence / math.log(2)).clamp(0, 1)


def class_topq(score, labels, eligible, coverage, classes=6):
    if not 0 <= coverage <= 1:
        raise ValueError("coverage must be in [0,1]")
    if score.shape != labels.shape or eligible.shape != labels.shape:
        raise ValueError("Score, labels and eligibility must share a grid")
    selected = torch.zeros_like(eligible, dtype=torch.bool)
    if coverage == 1:
        return eligible.bool() & (labels != 255)
    if coverage == 0:
        return selected
    flat = selected.flatten()
    for label in range(classes):
        indices = torch.nonzero((eligible.bool() & (labels == label)).flatten()).flatten()
        count = math.ceil(coverage * indices.numel())
        if count:
            # Stable ranking gives deterministic pixel-ID tie breaking.
            order = torch.argsort(score.flatten()[indices], descending=True, stable=True)
            flat[indices[order[:count]]] = True
    return selected


def selective_outer_loss(loss_map, weights, labels, pasted, eligibility, score, coverage, intervention=1.):
    if not math.isfinite(intervention) or not 0 <= intervention <= 1:
        raise ValueError('intervention must be in [0,1]')
    eligible = eligibility.bool() & (labels != 255) & (pasted == 255)
    selected = class_topq(score.detach(), labels.detach(), eligible, coverage)
    weighted = loss_map * weights
    if coverage == 1:
        loss = weighted.mean()
    else:
        count = eligible.sum()
        scale = count / selected.sum().clamp_min(1)
        # Keep donor/outside-overlap coefficients identical to the baseline.
        loss = (weighted.masked_fill(eligible, 0).sum()
                + scale * weighted.masked_fill(~selected, 0).sum()) / weighted.numel()
    if intervention != 1:
        loss = (1-intervention)*weighted.mean() + intervention*loss
    return loss, selected, eligible


def soft_js_outer_loss(loss_map, weights, labels, pasted, eligibility, divergence, strength=.5, penalty=1.):
    if not math.isfinite(strength) or not 0 <= strength <= 1 or not math.isfinite(penalty) or penalty < 0:
        raise ValueError('Invalid soft JS weighting')
    eligible = eligibility.bool() & (labels != 255) & (pasted == 255)
    # Suppress disagreement without a quota or amplification of selected pixels.
    coefficient = torch.where(eligible, 1-strength+strength*torch.exp(-penalty*divergence.detach()), 1.)
    return (loss_map*weights*coefficient).mean(), eligible, coefficient


class GSDSupervision:
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        root = Path(config["view_root"])
        self.root = root if root.is_absolute() else runtime.root / root
        path = self.root / "view_manifest.json"
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest["bands"] != config["target_bands"]:
            raise ValueError("GSD view bands do not match the study")
        self.records = self.manifest["records"]
        self.last_record = {}
        self.last_score = self.last_valid = self.last_divergence = None
        self.extra_forwards = 0

    def _load(self, names, view):
        images, valid, geometry = [], [], []
        from scripts.snd_remote import normalize
        for name in names:
            item = self.records[name]["views"][view]
            with Image.open(self.root / item["image"]) as image:
                images.append(normalize(image.convert("RGB")))
            with Image.open(self.root / item["valid"]) as mask:
                valid.append(torch.from_numpy(np.asarray(mask).copy() > 0))
            geometry.append(item["geometry"])
        return (torch.stack(images).to(self.runtime.device),
                torch.stack(valid).to(self.runtime.device), geometry)

    @torch.no_grad()
    def prepare(self, names, parameters, native_probability, encoder, classifier):
        valid = []
        for name in names:
            with Image.open(self.root / self.records[name]["common_native_valid"]) as mask:
                valid.append(torch.from_numpy(np.asarray(mask).copy() > 0))
        valid = torch.stack(valid).to(self.runtime.device)
        valid = self.runtime.full2weak(valid[:, None].float(), parameters)[:, 0] >= 1 - 1e-5
        confidence = native_probability.max(1).values
        arm = self.config["arm"]
        divergence = torch.zeros_like(confidence)
        if arm in ("generic", "gsd"):
            view = self.config["generic_view"] if arm == "generic" else self.config["gsd_view"]
            images, support, geometry = self._load(names, view)
            modes = encoder.training, classifier.training
            encoder.eval()
            classifier.eval()
            try:
                prediction = classifier(encoder(images)[1], images.shape[-2:])
            finally:
                encoder.train(modes[0])
                classifier.train(modes[1])
            # Match SND's native logits->warp->softmax order on both branches.
            aligned, overlap = warp_to_native(prediction, geometry, support, normalize=False)
            aligned = self.runtime.full2weak(aligned, parameters).softmax(1)
            overlap = self.runtime.full2weak(overlap[:, None].float(), parameters)[:, 0] >= 1 - 1e-5
            if (valid & ~overlap).any():
                raise ValueError("Cached common support differs from probability-warp support")
            divergence = normalized_js(native_probability, aligned)
            self.extra_forwards += len(names)
        self.last_score = confidence * torch.exp(-self.config["lambda"] * divergence)
        self.last_divergence = divergence
        self.last_valid = valid
        self.last_record = {"gsd_arm": arm, "gsd_extra_teacher_images": self.extra_forwards,
                            "gsd_js_mean": float(divergence[valid].mean()) if valid.any() else None}

    def loss(self, loss_map, weights, labels, pasted):
        mode = self.config.get('loss_mode', 'hard')
        strength = self.config.get('intervention', 1.)
        if mode == 'soft_js':
            loss, eligible, coefficient = soft_js_outer_loss(
                loss_map, weights, labels, pasted, self.last_valid, self.last_divergence,
                strength=strength, penalty=self.config['lambda'])
            selected = eligible
        else:
            loss, selected, eligible = selective_outer_loss(
                loss_map, weights, labels, pasted, self.last_valid,
                self.last_score, self.config["coverage"], intervention=strength)
            scale = eligible.sum()/selected.sum().clamp_min(1)
            coefficient = torch.where(eligible, 1-strength+strength*scale*selected, 1.)
        self.last_record.update(
            gsd_loss_mode=mode, gsd_intervention=strength,
            gsd_recipient_coefficient_mean=float(coefficient[eligible].mean()) if eligible.any() else None,
            gsd_recipient_coefficient_min=float(coefficient[eligible].min()) if eligible.any() else None,
            gsd_recipient_coefficient_max=float(coefficient[eligible].max()) if eligible.any() else None,
            gsd_eligible_fraction=float(eligible.float().mean()),
            gsd_selected_fraction=float(selected.float().mean()),
            gsd_recipient_selected_counts=torch.bincount(labels[selected], minlength=6).cpu().tolist(),
            gsd_recipient_eligible_counts=torch.bincount(labels[eligible], minlength=6).cpu().tolist(),
            gsd_actual_recipient_coverage=float(selected.sum() / eligible.sum().clamp_min(1)))
        return loss
