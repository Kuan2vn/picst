"""Phenology-interventional self-training (PIC-ST) hooks for the B2/IRRG DT-ST trainer.

C0  teacher-side: add delta to the teacher tree logit before pseudo-labels / negative selection.
    delta = median drop of the teacher tree margin when the leaf-off counterfactual
    x' = mu_ground + a*(x - mu_tree) (a=0.5, 65 px local means) is applied to pixels the
    teacher itself calls tree. Label-free; re-estimated every `update_every` updates on the
    unmixed target batch; pooled over the last `buffer` batches.
C1  student-side: with probability `c1_prob` per image, the same operator with a~U[a_min,1]
    is applied inside teacher-tree (incl. pasted tree donors) regions of the student input,
    labels unchanged. Ground means fall back to a 257 px window when the 65 px one is empty
    (crown interiors); edges are feathered.
null: enabled with c0=false, c1=false -> identical pseudo-labels/inputs to raw DT-ST.
Ablation-only optional keys (absent -> behaviour identical to the 31/10 HPC code):
  delta_schedule  [[iteration, delta], ...] piecewise-constant teacher tree offset replayed without C0
                  estimation (c0=false, fixed_delta=null); holds the last value after the final entry.
  c1_operator     'leaf_off' (default) or 'random_direction': same mask, texture factor a and per-pixel
                  mean-shift magnitude |mu_ground - mu_tree|, but a random spectral direction per image.
Opt-in extensions (09/11/2026, IRRG->RGB study; absent -> behaviour identical):
  c0_classes      [[class, [ground classes]], ...]: C0 on every G-selected vegetation class, each with its own
                  label-free delta (same operator/estimator, margin of that class). [[3, [0, 2, 5]]] == tree-only C0.
  c1_plausible    true: C1 keeps an edited pixel only where the teacher still predicts a vegetation class
                  (tree or low vegetation) on the edited student input (per-pixel, label-free plausibility).
See reports/rs_phenology_diagnosis_20261031/.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

TREE, GROUND = 3, (0, 2, 5)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
KEYS = {'c0', 'c1', 'c0_a', 'update_every', 'buffer', 'c1_prob', 'a_min', 'protect_pasted_tree', 'fixed_delta'}
OPTIONAL = {'delta_schedule', 'c1_operator', 'c0_classes', 'c1_plausible'}
VEGETATION = (2, 3)


def validate_phenology(settings):
    if not KEYS <= set(settings) <= KEYS | OPTIONAL:
        raise ValueError(f'Phenology settings must have {sorted(KEYS)} plus optionally {sorted(OPTIONAL)}')
    if type(settings['c0']) is not bool or type(settings['c1']) is not bool or type(settings['protect_pasted_tree']) is not bool:
        raise ValueError('c0/c1/protect_pasted_tree must be boolean')
    if not (0 < settings['c0_a'] <= 1 and 0 < settings['a_min'] <= 1 and 0 <= settings['c1_prob'] <= 1):
        raise ValueError('Invalid operator amplitudes/probability')
    if settings['update_every'] < 1 or settings['buffer'] < 1:
        raise ValueError('Invalid estimator schedule')
    if settings['fixed_delta'] is not None and settings['c0']:
        raise ValueError('fixed_delta is only for null/diagnostic arms with c0=false')
    schedule = settings.get('delta_schedule')
    if schedule is not None:
        steps = [point[0] for point in schedule]
        if settings['c0'] or settings['fixed_delta'] is not None or not steps or steps[0] != 0 or steps != sorted(set(steps)):
            raise ValueError('delta_schedule needs c0=false, fixed_delta=null and strictly increasing iterations from 0')
    if settings.get('c1_operator', 'leaf_off') not in ('leaf_off', 'random_direction'):
        raise ValueError('Unknown c1_operator')
    classes = settings.get('c0_classes')
    if classes is not None:
        if (not settings['c0'] or settings['fixed_delta'] is not None or schedule is not None or not classes or
                len({c for c, _ in classes}) != len(classes) or
                any(not 0 <= c < 6 or not ground or c in ground or not all(0 <= g < 6 for g in ground) for c, ground in classes)):
            raise ValueError('c0_classes needs c0=true and distinct classes with disjoint non-empty ground sets')
    if type(settings.get('c1_plausible', False)) is not bool or (settings.get('c1_plausible') and not settings['c1']):
        raise ValueError('c1_plausible must be boolean and needs c1=true')
    return settings


def _pixels(t):
    return (t * STD.to(t.device) + MEAN.to(t.device)) * 255.0


def _normalised(x):
    return (x / 255.0 - MEAN.to(x.device)) / STD.to(x.device)


def _local_mean(x, mask, k):
    m = mask.float()[:, None]
    pool = lambda v: F.avg_pool2d(v, k, stride=1, padding=k // 2, count_include_pad=True)  # noqa: E731
    count = pool(m) * k * k
    return pool(x * m) * k * k / count.clamp_min(1), count[:, 0]


def leaf_off(x, tree, ground, a, fallback=True, direction=None):
    """x: B,3,H,W pixel values; tree/ground: B,H,W bool; a: B,1,1,1. Returns edited x and edit mask."""
    mu_g, n_g = _local_mean(x, ground, 65)
    if fallback:
        mu_g2, n_g2 = _local_mean(x, ground, 257)
        use2 = (n_g <= 50) & (n_g2 > 200)
        mu_g = torch.where(use2[:, None], mu_g2, mu_g)
        n_g = torch.where(use2, n_g2, n_g)
    mu_t, _ = _local_mean(x, tree, 65)
    ok = tree & (n_g > 50)
    if direction is None:
        edited = mu_g + a * (x - mu_t)
    else:  # random-direction control: same per-pixel shift magnitude and texture factor
        edited = mu_t + (mu_g - mu_t).norm(dim=1, keepdim=True) * direction + a * (x - mu_t)
    return torch.where(ok[:, None], edited, x), ok


def _margin(z, c=TREE):
    other = torch.cat([z[:, :c], z[:, c + 1:]], 1).max(1).values
    return z[:, c] - other


class PhenologyCorrection:
    def __init__(self, runtime, settings):
        self.s = settings
        self.c0, self.c1 = settings['c0'], settings['c1']
        self.drops = deque(maxlen=settings['buffer'])
        self.delta = 0.0 if settings['fixed_delta'] is None else float(settings['fixed_delta'])
        self.output = Path(getattr(runtime, 'output', '.'))
        self.state_path = self.output / 'phenology_state.json'
        self.log_path = self.output / 'phenology_log.jsonl'
        self.last_update = None
        self.c1_pixels = self.c1_images = 0
        self.multi = settings.get('c0_classes')
        self.c1_rejected = 0
        self.teacher = None
        if self.multi is not None:
            self.deltas = {c: 0.0 for c, _ in self.multi}
            self.multi_drops = {c: deque(maxlen=settings['buffer']) for c, _ in self.multi}
        if self.state_path.exists():  # resume: restore the pooled estimator
            state = json.loads(self.state_path.read_text())
            self.delta = state['delta']; self.last_update = state['last_update']
            self.drops.extend(torch.tensor(v) for v in state['drops'])
            if self.multi is not None:
                self.deltas = {int(c): v for c, v in state['deltas'].items()}
                for c, values in state['multi_drops'].items():
                    self.multi_drops[int(c)].extend(torch.tensor(v) for v in values)

    @torch.no_grad()
    def _estimate_multi(self, images, encoder, classifier):
        x = _pixels(images)
        logits = classifier(encoder(images)[1], images.shape[-2:])
        pred = logits.argmax(1)
        a = torch.full((x.shape[0], 1, 1, 1), self.s['c0_a'], device=x.device)
        for c, ground_classes in self.multi:
            ground = torch.zeros_like(pred, dtype=torch.bool)
            for g in ground_classes:
                ground |= pred == g
            edited, ok = leaf_off(x, pred == c, ground, a, fallback=False)
            if ok.sum() < 200:
                continue
            z1 = classifier(encoder(_normalised(edited))[1], images.shape[-2:])
            self.multi_drops[c].append((_margin(logits, c) - _margin(z1, c))[ok][::16].float().cpu())
            self.deltas[c] = float(torch.cat(list(self.multi_drops[c])).median())
        self.delta = self.deltas.get(TREE, 0.0)

    @torch.no_grad()
    def _estimate(self, images, logits, encoder, classifier):
        x = _pixels(images)
        logits = classifier(encoder(images)[1], images.shape[-2:])  # eval-mode reference margin
        pred = logits.argmax(1)
        tree = pred == TREE
        ground = torch.zeros_like(tree)
        for c in GROUND:
            ground |= pred == c
        edited, ok = leaf_off(x, tree, ground, torch.full((x.shape[0], 1, 1, 1), self.s['c0_a'], device=x.device), fallback=False)
        if ok.sum() < 200:
            return
        z1 = classifier(encoder(_normalised(edited))[1], images.shape[-2:])
        drop = (_margin(logits) - _margin(z1))[ok][::16].float().cpu()
        self.drops.append(drop)
        self.delta = float(torch.cat(list(self.drops)).median())

    def adjust(self, teacher_logits, images, encoder, classifier, iteration):
        """Return teacher logits with the C0 offset on the tree channel (no-op when c0 is off)."""
        if self.c0 and (self.last_update is None or iteration - self.last_update >= self.s['update_every']):
            modes = [(m, m.training) for model in (encoder, classifier) for m in model.modules()]
            try:
                encoder.eval(); classifier.eval()
                if self.multi is not None:
                    self._estimate_multi(images, encoder, classifier)
                else:
                    self._estimate(images, teacher_logits.detach(), encoder, classifier)
            finally:
                for m, mode in modes:
                    m.training = mode
            self.last_update = iteration
            self._save(iteration)
        schedule = self.s.get('delta_schedule')
        if schedule is not None:
            self.delta = float(next(d for step, d in reversed(schedule) if step <= iteration))
        self.teacher = (encoder, classifier)
        if self.multi is not None:
            out = teacher_logits.clone()
            for c, value in self.deltas.items():
                out[:, c] += value
            return out
        if self.delta == 0:
            return teacher_logits
        out = teacher_logits.clone(); out[:, TREE] += self.delta
        return out

    @torch.no_grad()
    def counterfactual(self, student_input, aligned_teacher_logits, mix_label, valid):
        """C1: leaf-off edit inside teacher-tree (and pasted tree) regions of the student input."""
        if not self.c1:
            return student_input
        x = _pixels(student_input)
        pred = aligned_teacher_logits.argmax(1)
        pasted = mix_label != 255
        tree = ((pred == TREE) & ~pasted) | (mix_label == TREE)
        ground = torch.zeros_like(tree)
        for c in GROUND:
            ground |= pred == c
        ground &= ~pasted & valid
        b = x.shape[0]
        a = torch.empty(b, 1, 1, 1, device=x.device).uniform_(self.s['a_min'], 1.0)
        apply = (torch.rand(b, device=x.device) < self.s['c1_prob'])[:, None, None]
        direction = None
        if self.s.get('c1_operator', 'leaf_off') == 'random_direction':
            direction = F.normalize(torch.randn(b, 3, 1, 1, device=x.device), dim=1)
        edited, ok = leaf_off(x, tree & valid, ground, a, direction=direction)
        ok &= apply
        if self.s.get('c1_plausible') and self.teacher is not None and bool(ok.any()):
            encoder, classifier = self.teacher
            modes = [(m, m.training) for model in (encoder, classifier) for m in model.modules()]
            try:
                encoder.eval(); classifier.eval()
                trial = _normalised((x + (edited - x) * ok[:, None].float()).clamp(0, 255))
                pred_edit = classifier(encoder(trial)[1], trial.shape[-2:]).argmax(1)
            finally:
                for m, mode in modes:
                    m.training = mode
            plausible = torch.zeros_like(ok)
            for c in VEGETATION:
                plausible |= pred_edit == c
            self.c1_rejected += int((ok & ~plausible).sum())
            ok &= plausible
        w = F.avg_pool2d(ok.float()[:, None], 7, stride=1, padding=3) * ok.float()[:, None]  # feather inside edges
        self.c1_pixels += int(ok.sum()); self.c1_images += int(apply.sum())
        return _normalised((x + w * (edited - x)).clamp(0, 255))

    def protected_labels(self, mix_label):
        if not self.s['protect_pasted_tree']:
            return None
        return torch.where(mix_label == TREE, mix_label, torch.full_like(mix_label, 255))

    def _save(self, iteration):
        state = dict(delta=self.delta, last_update=self.last_update, drops=[d.tolist() for d in self.drops])
        if self.multi is not None:
            state.update(deltas={str(c): v for c, v in self.deltas.items()},
                         multi_drops={str(c): [d.tolist() for d in q] for c, q in self.multi_drops.items()})
        tmp = self.state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state)); tmp.replace(self.state_path)
        with self.log_path.open('a') as f:
            record = dict(iteration=iteration, delta=self.delta, pooled=int(sum(len(d) for d in self.drops)),
                          c1_pixels=self.c1_pixels, c1_images=self.c1_images)
            if self.multi is not None:
                record['deltas'] = {str(c): v for c, v in self.deltas.items()}
            if self.s.get('c1_plausible'):
                record['c1_rejected'] = self.c1_rejected
            f.write(json.dumps(record) + '\n')
