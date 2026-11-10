"""Re-implementations (not reproductions) of two remote-sensing SFDA baselines on the shared SegFormer-B2 host.

Both run inside the same B2 trainer as DT-ST / PIC-ST (train_SND.py through train_DTST_remote.py), with the
same source checkpoint, data, batch 6, AdamW recipe, evaluation and stopping rule. Only the adaptation loss and
the pseudo-label rule are replaced, through the opt-in ``runtime.baseline`` hook; without a baseline config the
host path is unchanged.

S3T-SFDA  Xi et al., "Self-Supervised Spatial-Temporal Consistency for Source-Free Domain-Adaptive
          Segmentation in Remote Sensing Images", IEEE TGRS 2025/26 (DOI 10.1109/TGRS.2025.3643744).
          Follows the released code (github.com/yixinzs/S3T-SFDA, MVCSFDAEncoderDecoder) where it is concrete:
          DT-ST host (dynamic teacher, SND proxy, TCR paste, negative BCE weight 1), spatial views by
          roi_align of 4 random square boxes per image (side 0.5-1 of the crop, p = 0.5) with roi_pool-ed
          teacher labels (SMVC), historical-agreement weights 2 / 0.5 with the student snapshot refreshed every
          300 iterations (TDC as released), and the historical pixel contrast with a conv-MLP projection head on
          the 4 concatenated MiT stages, reliability from the historical entropy, weight 0.5 (STC).
          Disclosed differences: the paper's KL-based TDC weight and the entropy term L_E are not in the released
          loss and are not used; the teacher EMA covers the whole network (paper Eq. 3) whereas the released
          code moves only the decode head.
St-CGN    Zhang et al., "St-CGN: Multiscale Consistency and Dual-View Entropy Fusion for Source-Free Domain
          Adaptation in Remote Sensing Semantic Segmentation", IEEE TGRS 64 (2026) 5640816. No public code.
          ST stage here: CbPCM (prototype-calibrated, class-balanced pseudo-labels, tau_cal 0.1, p_ratio
          0.4 -> 0.9) and DvPEF (KL teacher/student + marginal-entropy regulariser, lambda_ent 0.1,
          lambda_kld 0.9), mean teacher EMA 0.996 (set in the host settings), no paste, no negative learning.
          The warm-up stage (MsFCL on labelled source) is train_stcgn_warmup.py; anchors come from it.
          ASGCF is a DeepLabV3+ decoder module and is not used: every method shares the B2 architecture.
"""
from __future__ import annotations

import copy
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

C = 6
METHODS = ('s3t_sfda', 'stcgn')


def validate_baseline(settings):
    method = settings.get('method')
    if method == 's3t_sfda':
        keys = {'method', 'roi_prob', 'roi_num', 'roi_scale', 'roi_size', 'history_every', 'agree_weight',
                'disagree_weight', 'contrast_weight', 'contrast_temperature', 'proj_dim', 'proj_lr_mult'}
    elif method == 'stcgn':
        keys = {'method', 'tau_cal', 'p_ratio_start', 'p_ratio_end', 'p_ratio_iters', 'lambda_ent',
                'prototype_every', 'prototypes'}
    else:
        raise ValueError(f'Unknown baseline method {method!r}')
    if set(settings) != keys:
        raise ValueError(f'{method} settings must have exactly {sorted(keys)}')
    if method == 's3t_sfda':
        lo, hi = settings['roi_scale']
        if not (0 <= settings['roi_prob'] <= 1 and settings['roi_num'] >= 1 and 0 < lo <= hi <= 1 and
                settings['history_every'] >= 1 and settings['contrast_weight'] >= 0):
            raise ValueError('Invalid S3T-SFDA settings')
    else:
        if not (0 < settings['tau_cal'] and 0 < settings['p_ratio_start'] <= settings['p_ratio_end'] <= 1 and
                settings['p_ratio_iters'] >= 1 and 0 <= settings['lambda_ent'] <= 1 and settings['prototype_every'] >= 1):
            raise ValueError('Invalid St-CGN settings')
    return settings


def build_baseline(runtime, settings, device):
    validate_baseline(settings)
    return (S3TSFDA if settings['method'] == 's3t_sfda' else StCGN)(runtime, settings, device)


def _eval_mode(*models):
    modes = [(m, m.training) for model in models for m in model.modules()]
    for model in models:
        model.eval()
    return modes


def _restore(modes):
    for m, mode in modes:
        m.training = mode


class _Hook:
    needs_unmixed = False

    def __init__(self, runtime, settings, device):
        self.runtime, self.s, self.device = runtime, settings, torch.device(device)
        self.last_record = {}

    def step(self, lr):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


# ----------------------------------------------------------------------------------------------- S3T-SFDA
class ProjectionHead(nn.Module):
    """Released S3T 'convmlp' head: 1x1 conv, BN, ReLU, 1x1 conv, L2 normalisation."""

    def __init__(self, dim_in, proj_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(dim_in, dim_in, 1), nn.BatchNorm2d(dim_in), nn.ReLU(),
                                  nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


def _concat_levels(features):
    size = features[1].shape[-2:]
    return torch.cat([F.interpolate(f, size=size, mode='bilinear', align_corners=True) for f in features], 1)


def historical_contrast(emb, labels, his_emb, his_logits, temperature=0.07):
    """Released Contrast_loss_his with update_contrast_his inputs, computed per image (same mean)."""
    b, _, h, w = emb.shape
    labels = F.interpolate(labels[:, None].float(), size=(h, w), mode='nearest').view(b, -1)
    prob = his_logits.softmax(1)
    ent = -(prob * torch.log2(prob + 1e-30))
    ent = F.interpolate(ent, size=(h, w), mode='bilinear', align_corners=True).view(b, ent.shape[1], -1).sum(1)
    reli = torch.clamp(1 - ent / math.log2(his_logits.shape[1]) + .1, min=.9, max=1.1)
    a = F.normalize(emb.view(b, emb.shape[1], -1), dim=1).transpose(1, 2)
    k = F.normalize(his_emb.view(b, his_emb.shape[1], -1), dim=1).transpose(1, 2)
    n = a.shape[1]
    losses = []
    for i in range(b):
        mask = (labels[i][:, None] == labels[i][None, :]).float()
        logits = a[i] @ k[i].T / temperature
        logits = logits - logits.max(1, keepdim=True).values.detach()
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        rel = reli[i][None, :].expand(n, n)
        exp_logits = torch.exp(logits) * logits_mask * rel
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20) + torch.log(rel + 1e-20)
        losses.append(-((mask * log_prob).sum(1) / n).mean())
    return torch.stack(losses).mean()


class S3TSFDA(_Hook):
    def __init__(self, runtime, settings, device):
        super().__init__(runtime, settings, device)
        self.proj = ProjectionHead(64 + 128 + 320 + 512, settings['proj_dim']).to(self.device)
        self.optimizer = torch.optim.AdamW(self.proj.parameters(), lr=1e-5, weight_decay=.01)
        self.his = None  # (encoder, classifier) student snapshot
        self.snapshot_iteration = None
        self.roi_iterations = 0

    def _boxes(self, batch, size):
        lo, hi = self.s['roi_scale']
        boxes = []
        for _ in range(batch):
            side = int(size * (lo + (hi - lo) * torch.rand(()).item()))
            rows = []
            for _ in range(self.s['roi_num']):
                y = int(torch.randint(0, size - side + 1, ()).item()); x = int(torch.randint(0, size - side + 1, ()).item())
                rows.append([x, y, x + side, y + side])
            boxes.append(torch.tensor(rows, dtype=torch.float32, device=self.device))
        return boxes

    def _refresh_history(self, iteration, encoder, classifier):
        if self.his is None or iteration % self.s['history_every'] == 0 and self.snapshot_iteration != iteration:
            enc, cls = copy.deepcopy(encoder), copy.deepcopy(classifier)
            for p in list(enc.parameters()) + list(cls.parameters()):
                p.requires_grad_(False)
            enc.eval(); cls.eval()
            self.his, self.snapshot_iteration = (enc, cls), iteration

    def loss(self, *, iteration, encoder, classifier, student_input, teacher_prob, pseudo, valid, criterion,
             negative_weight, **_):
        from torchvision.ops import roi_align, roi_pool
        from scripts.dtst_remote import negative_loss
        self._refresh_history(iteration, encoder, classifier)
        x, labels, prob, valid = student_input, pseudo.long(), teacher_prob, valid.bool()
        roi = torch.rand(()).item() < self.s['roi_prob']
        if roi:
            out = (self.s['roi_size'], self.s['roi_size'])
            boxes = self._boxes(x.shape[0], x.shape[-1])
            x = roi_align(x, boxes, out)
            labels = roi_pool(labels[:, None].float(), boxes, out)[:, 0].long()
            prob = roi_pool(prob, boxes, out)
            valid = roi_pool(valid[:, None].float(), boxes, out)[:, 0] > .5
            self.roi_iterations += 1
        features = encoder(x)[1]
        logits = classifier(features, x.shape[-2:])
        emb = self.proj(_concat_levels(features))
        with torch.no_grad():
            his_enc, his_cls = self.his
            his_features = his_enc(x)[1]
            his_logits = his_cls(his_features, x.shape[-2:])
            his_emb = self.proj(_concat_levels(his_features))
        weight = torch.full_like(labels, self.s['disagree_weight'], dtype=torch.float32)
        weight[his_logits.argmax(1) == labels] = self.s['agree_weight']
        positive = (criterion(logits, labels) * weight).mean()
        negative = negative_loss(logits, prob, valid, negative_weight)
        contrast = historical_contrast(emb, labels, his_emb, his_logits, self.s['contrast_temperature'])
        total = positive + negative + self.s['contrast_weight'] * contrast
        self.last_record = {'s3t_roi': roi, 's3t_roi_iterations': self.roi_iterations,
                            's3t_contrast': float(contrast.detach()), 's3t_history_iteration': self.snapshot_iteration,
                            's3t_agreement': float((his_logits.argmax(1) == labels)[labels != 255].float().mean())
                            if (labels != 255).any() else None}
        return total, positive, negative

    def step(self, lr):
        for group in self.optimizer.param_groups:
            group['lr'] = lr * self.s['proj_lr_mult']
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {'proj': self.proj.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'his': None if self.his is None else (self.his[0].state_dict(), self.his[1].state_dict()),
                'snapshot_iteration': self.snapshot_iteration, 'roi_iterations': self.roi_iterations}

    def load_state_dict(self, state, encoder=None, classifier=None):
        self.proj.load_state_dict(state['proj']); self.optimizer.load_state_dict(state['optimizer'])
        self.snapshot_iteration, self.roi_iterations = state['snapshot_iteration'], state['roi_iterations']
        if state['his'] is not None:
            enc, cls = copy.deepcopy(encoder), copy.deepcopy(classifier)
            enc.load_state_dict(state['his'][0]); cls.load_state_dict(state['his'][1])
            for p in list(enc.parameters()) + list(cls.parameters()):
                p.requires_grad_(False)
            enc.eval(); cls.eval()
            self.his = (enc, cls)


# -------------------------------------------------------------------------------------------------- St-CGN
def class_balanced_labels(prob, rectified, valid, p_ratio, ignore=255):
    """CbPCM Eq. 12: k = argmax rectified; keep if the ORIGINAL probability of k reaches the class threshold,
    the value ranked at the top p_ratio of the original predictions of class k in the mini-batch."""
    k = rectified.argmax(1)
    original = prob.gather(1, k[:, None])[:, 0]
    argmax = prob.argmax(1)
    thresholds = torch.full((prob.shape[1],), float('inf'), device=prob.device)
    for c in range(prob.shape[1]):
        values = prob[:, c][(argmax == c) & valid]
        if values.numel():
            thresholds[c] = torch.quantile(values.float(), 1 - p_ratio) if values.numel() > 1 else values[0]
    keep = valid & (original >= thresholds[k])
    return torch.where(keep, k, torch.full_like(k, ignore)), thresholds


class StCGN(_Hook):
    needs_unmixed = True

    def __init__(self, runtime, settings, device):
        super().__init__(runtime, settings, device)
        anchors = torch.load(settings['prototypes'], map_location='cpu', weights_only=False)
        self.prototypes = anchors['prototypes'].float().to(self.device)  # C x D
        self.sums = torch.zeros_like(self.prototypes)
        self.counts = torch.zeros(self.prototypes.shape[0], device=self.device)
        self.refreshes = 0

    def p_ratio(self, iteration):
        s = self.s
        t = min(1.0, iteration / s['p_ratio_iters'])
        return s['p_ratio_start'] + (s['p_ratio_end'] - s['p_ratio_start']) * t

    def loss(self, *, iteration, encoder, classifier, unmixed, full, params, teacher_prob, valid, criterion, **_):
        runtime = self.runtime
        valid = valid.bool()
        with torch.no_grad():  # student evidence on the weak (full) view, aligned to the student crop
            modes = _eval_mode(encoder, classifier)
            try:
                features = encoder(full)[1]
                fused = classifier.fuse_features(features)
                full_logits = classifier.classify_fused(fused, full.shape[-2:])
            finally:
                _restore(modes)
            prob = runtime.full2weak(full_logits, params).softmax(1)
            z = F.normalize(runtime.full2weak(fused, params, down_ratio=4), dim=1)
            similarity = torch.einsum('bdhw,kd->bkhw', z, F.normalize(self.prototypes, dim=1))
            semantic = (similarity / self.s['tau_cal']).softmax(1)
            semantic = F.interpolate(semantic, size=prob.shape[-2:], mode='bilinear', align_corners=False)
            rectified = prob * semantic
            rectified = rectified / rectified.sum(1, keepdim=True).clamp_min(1e-12)
            ratio = self.p_ratio(iteration)
            labels, thresholds = class_balanced_labels(prob, rectified, valid, ratio)
            # prototype refresh evidence (target, accepted pseudo-labels, same feature space as the anchors)
            small = F.interpolate(labels[:, None].float(), size=z.shape[-2:], mode='nearest')[:, 0].long()
            raw = runtime.full2weak(fused, params, down_ratio=4)
            for c in range(self.prototypes.shape[0]):
                m = small == c
                if m.any():
                    self.sums[c] += raw.permute(0, 2, 3, 1)[m].sum(0)
                    self.counts[c] += m.sum()
        logits = classifier(encoder(unmixed)[1], unmixed.shape[-2:])
        log_ps = F.log_softmax(logits, 1)
        ps = log_ps.exp()
        pt = teacher_prob
        target = criterion(logits, labels).mean()
        v = valid[:, None].float()
        kld = ((pt * (torch.log(pt.clamp_min(1e-12)) - log_ps)) * v).sum() / v.sum().clamp_min(1)
        unified = .5 * (pt + ps)
        marginal = (unified * v).sum((0, 2, 3)) / v.sum().clamp_min(1)
        ent = (marginal * torch.log(marginal.clamp_min(1e-12))).sum()
        lam = self.s['lambda_ent']
        total = target + lam * ent + (1 - lam) * kld
        if (iteration + 1) % self.s['prototype_every'] == 0:
            present = self.counts > 0
            self.prototypes[present] = self.sums[present] / self.counts[present, None]
            self.sums.zero_(); self.counts.zero_(); self.refreshes += 1
        self.last_record = {'stcgn_p_ratio': ratio, 'stcgn_kld': float(kld.detach()), 'stcgn_ent': float(ent.detach()),
                            'stcgn_label_fraction': float((labels != 255).float().mean()),
                            'stcgn_prototype_refreshes': self.refreshes,
                            'stcgn_thresholds': [None if math.isinf(t) else round(float(t), 4) for t in thresholds]}
        return total, target, torch.zeros((), device=logits.device)

    def state_dict(self):
        return {'prototypes': self.prototypes.cpu(), 'sums': self.sums.cpu(), 'counts': self.counts.cpu(),
                'refreshes': self.refreshes}

    def load_state_dict(self, state, encoder=None, classifier=None):
        self.prototypes = state['prototypes'].to(self.device); self.sums = state['sums'].to(self.device)
        self.counts = state['counts'].to(self.device); self.refreshes = state['refreshes']


def load_settings(path):
    with open(path, encoding='utf-8') as handle:
        return validate_baseline(json.load(handle))
