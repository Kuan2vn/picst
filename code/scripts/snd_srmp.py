"""Masked source-relational auxiliary loss for the B2 SND continuation."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from train_SRMP import SpatialRebuilder, align_full_image, masked_mean, relational_cosine_map, sample_block_mask


def source_readout(source_state):
    head = source_state['classifier']
    weight = head['linear_pred.weight'].detach().float().clone()
    bias = head['linear_pred.bias'].detach().float().clone()
    if weight.shape != (6, 768, 1, 1) or bias.shape != (6,):
        raise ValueError('Expected six-class 768-channel source Conv1x1 readout')
    rows = weight.flatten(1)
    norms = rows.norm(dim=1, keepdim=True)
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all() or (norms <= 1e-12).any():
        raise ValueError('Source readout has non-finite or zero class rows')
    directions = rows / norms
    gram = directions @ directions.T
    return weight, bias, directions, (gram + gram.T) / 2


@torch.no_grad()
def clean_student_target(student, decoder, image):
    """Use current weights without dropout, BN updates or a target-side graph."""
    modes = [(module, module.training) for model in (student, decoder) for module in model.modules()]
    try:
        student.eval()
        decoder.eval()
        return decoder.fuse_features(student(image)[1]).detach()
    finally:
        # Preserve individually frozen BN/dropout modes, not just the root flags.
        for module, training in modes:
            module.training = training


def gradient_alignment(main_grads, aux_grads):
    """Norm ratio and cosine on the same probe parameters; zero cosine is undefined."""
    main_sq = sum(g.square().sum() for g in main_grads)
    aux_sq = sum(g.square().sum() for g in aux_grads)
    denominator = (main_sq * aux_sq).sqrt()
    ratio = float((aux_sq.sqrt() / main_sq.sqrt().clamp_min(1e-12)).detach())
    cosine = (float((sum((m * a).sum() for m, a in zip(main_grads, aux_grads)) /
                     denominator).clamp(-1, 1).detach()) if denominator > 1e-12 else None)
    return ratio, cosine


@torch.no_grad()
def teacher_site_mask(probabilities, sites, threshold=0., full_view_probabilities=None):
    """Select detached teacher targets; full/crop agreement uses the source head."""
    selected = sites
    if threshold:
        selected = selected & (probabilities.max(1, keepdim=True).values >= threshold)
    if full_view_probabilities is not None:
        if full_view_probabilities.shape != probabilities.shape:
            raise ValueError('Full/crop teacher probabilities must share the same grid')
        selected = selected & (probabilities.argmax(1, keepdim=True) ==
                               full_view_probabilities.argmax(1, keepdim=True))
    return selected


def relational_with_soft_kl(logits, teacher_prob, gram, kl_weight):
    """Source Gram cosine plus soft-target KL with a stable logit gradient."""
    target = teacher_prob.detach().float()
    relation = relational_cosine_map(logits.softmax(1), target, gram)
    kl = (F.kl_div(logits.float().log_softmax(1), target, reduction='none').sum(1)
          if kl_weight else None)
    return relation if kl is None else relation + kl_weight * kl, relation, kl


def source_context_kernel(queries, keys, directions, gram, source_weight=1.):
    """Cosine in I + alpha U.T A U, without building a768x768 matrix."""
    u, a = directions.detach(), gram.detach()
    queries, keys = F.normalize(queries.float(), dim=-1, eps=1e-6), F.normalize(keys.float(), dim=-1, eps=1e-6)
    rq, rk = queries @ u.T, keys @ u.T
    numerator = queries @ keys.T + source_weight * (rq @ a @ rk.T)
    nq = (queries.square().sum(-1) + source_weight * (rq @ a * rq).sum(-1)).clamp_min(1e-12).sqrt()
    nk = (keys.square().sum(-1) + source_weight * (rk @ a * rk).sum(-1)).clamp_min(1e-12).sqrt()
    return numerator / (nq[:, None] * nk[None, :])


def context_relational_loss(rebuilt, teacher, hidden, pixel_valid, directions, gram,
                            temperature=.1, source_weight=1.):
    """Hidden queries recover their relationships to detached visible teacher keys."""
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(source_weight) or source_weight < 0:
        raise ValueError('Context temperature must be positive and source weight nonnegative')
    if rebuilt.shape != teacher.shape or hidden.shape[:2] != (rebuilt.shape[0], 1):
        raise ValueError('Context teacher/rebuilt features and hidden batch must match')
    grid = hidden.shape[-2:]
    zs = F.adaptive_avg_pool2d(rebuilt.float(), grid).flatten(2).transpose(1, 2)
    zt = F.adaptive_avg_pool2d(teacher.detach().float(), grid).flatten(2).transpose(1, 2)
    valid = (F.adaptive_max_pool2d((~pixel_valid.bool()).float(), grid) == 0).flatten(1)
    masks = hidden.bool().flatten(1)
    loss_sum = rebuilt.sum() * 0
    count = key_count = entropy_sum = info_sum = energy_sum = variance_sum = 0
    eligible = 0
    for b in range(rebuilt.shape[0]):
        queries, visible = masks[b] & valid[b], ~masks[b] & valid[b]
        n, k = int(queries.sum()), int(visible.sum())
        # One key yields a constant categorical distribution and no learning signal.
        if not n or k < 2:
            continue
        tq, keys = zt[b, queries], zt[b, visible].detach()
        target_log = (source_context_kernel(tq, keys, directions, gram, source_weight) / temperature).log_softmax(-1)
        target = target_log.exp().detach()
        predicted = (source_context_kernel(zs[b, queries], keys, directions, gram, source_weight) / temperature).log_softmax(-1)
        loss_sum = loss_sum + F.kl_div(predicted, target, reduction='sum')
        with torch.no_grad():
            entropy = -(target * target_log).sum(-1)
            normalized = F.normalize(tq, dim=-1, eps=1e-6)
            r = normalized @ directions.detach().T
            source_energy = source_weight * (r @ gram.detach() * r).sum(-1)
            energy_sum += float((source_energy / (normalized.square().sum(-1)+source_energy).clamp_min(1e-12)).sum())
            entropy_sum += float((entropy / math.log(k)).sum())
            info_sum += float((math.log(k) - entropy).clamp_min(0).sum())
            variance_sum += float(zt[b, valid[b]].var(dim=0, unbiased=False).mean())
        count += n; key_count += k; eligible += 1
    stats = dict(aux_selected_cells=count, context_visible_keys=key_count/max(eligible, 1),
        context_target_entropy=entropy_sum/max(count, 1), context_target_information=info_sum/max(count, 1),
        context_source_energy_fraction=energy_sum/max(count, 1), context_teacher_variance=variance_sum/max(eligible, 1),
        mask_fraction=float((masks & valid).sum()/valid.sum().clamp_min(1)),
        aux_selected_fraction=count/max(int((masks & valid).sum()), 1), context_grid=list(grid))
    return loss_sum/max(count, 1), stats


def masked_source_contrastive_loss(rebuilt, teacher, hidden, pixel_valid, weight, bias, gram,
                                   temperature=.1, semantic_temperature=2., grid=32,
                                   keys_per_class=32, generator=None):
    """Recover paired clean latents; source/target relations temper false negatives.

    Teacher argmax only stratifies the key pool. The positive target is the same
    spatial cell, never a hard class label. No student class softmax in the loss.
    """
    if (not math.isfinite(temperature) or temperature <= 0 or
            not math.isfinite(semantic_temperature) or semantic_temperature <= 0 or
            type(grid) is not int or grid < 1 or type(keys_per_class) is not int or keys_per_class < 1):
        raise ValueError('Invalid source contrast temperature/grid/key count')
    if rebuilt.shape != teacher.shape or hidden.shape[:2] != (rebuilt.shape[0], 1):
        raise ValueError('Contrast teacher/rebuilt shapes and hidden batch must match')
    zs = F.normalize(F.adaptive_avg_pool2d(rebuilt.float(), grid).permute(0, 2, 3, 1).reshape(-1, rebuilt.shape[1]), dim=1)
    with torch.no_grad():
        zt = F.normalize(F.adaptive_avg_pool2d(teacher.detach().float(), grid).permute(0, 2, 3, 1).reshape_as(zs), dim=1)
        prob = F.conv2d(teacher.detach().float(), weight.detach(), bias.detach()).div(semantic_temperature).softmax(1)
        prob = F.adaptive_avg_pool2d(prob, grid).permute(0, 2, 3, 1).reshape(-1, gram.shape[0])
        labels = prob.argmax(1)
        valid = (F.adaptive_max_pool2d((~pixel_valid.bool()).float(), grid) == 0).flatten()
        masked = F.interpolate(hidden.float(), size=(grid, grid), mode='nearest').bool().flatten() & valid
        query_ids = masked.nonzero().flatten()
        chosen = []
        for cls in range(gram.shape[0]):
            ids = (valid & (labels == cls)).nonzero().flatten()
            if len(ids):
                chosen.append(ids[torch.randperm(len(ids), device=ids.device, generator=generator)[:keys_per_class]])
        key_ids = torch.cat(chosen) if chosen else query_ids[:0]
    stats = dict(aux_selected_cells=0, aux_selected_fraction=0., contrast_grid=grid,
        contrast_keys=int(len(key_ids)), contrast_keys_by_class=torch.bincount(labels[key_ids], minlength=gram.shape[0]).tolist(),
        mask_fraction=float(masked.sum()/valid.sum().clamp_min(1)))
    if not len(query_ids) or not len(key_ids):
        return rebuilt.sum()*0, stats
    with torch.no_grad():
        q, k, a = prob[query_ids], prob[key_ids], gram.detach().float()
        qnorm = (q @ a*q).sum(1).clamp_min(1e-12).sqrt()
        knorm = (k @ a*k).sum(1).clamp_min(1e-12).sqrt()
        crs = ((q @ a @ k.T)/(qnorm[:, None]*knorm[None, :])).clamp(-1, 1)
        neg_weight = (1-crs).clamp(0, 2)
        # Exclude the paired key, and numerically identical semantic signatures.
        neg_weight[(query_ids[:, None] == key_ids[None, :]) | (neg_weight < 1e-4)] = 0
        source_mass = neg_weight.sum(1)
        # Clean geometry can remain useful when the source readout misassigns a class.
        neg_weight = neg_weight * (1-(zt[query_ids] @ zt[key_ids].T).clamp(-1, 1))
        eligible = neg_weight.sum(1) > 0
    if not eligible.any():
        return rebuilt.sum()*0, stats
    ids = query_ids[eligible]
    neg_weight = neg_weight[eligible]
    positive = (zs[ids]*zt[ids]).sum(1)
    negative = zs[ids] @ zt[key_ids].T
    # log(1 + sum_j w_ij exp((sim_ij - sim_ii)/tau)), stable even at large margins.
    neg_log = ((negative-positive[:, None])/temperature +
               neg_weight.clamp_min(1e-30).log()).masked_fill(neg_weight == 0, -torch.inf)
    loss = torch.logsumexp(torch.cat((torch.zeros_like(positive[:, None]), neg_log), 1), 1).mean()
    with torch.no_grad():
        hardest = negative.masked_fill(neg_weight == 0, -torch.inf).max(1).values
        stats.update(aux_selected_cells=int(len(ids)), aux_selected_fraction=float(len(ids)/len(query_ids)),
            contrast_queries_by_class=torch.bincount(labels[ids], minlength=gram.shape[0]).tolist(),
            contrast_positive_cosine=float(positive.mean()),
            contrast_negative_cosine=float((negative*neg_weight).sum()/neg_weight.sum()),
            contrast_positive_margin=float((positive-hardest).mean()),
            contrast_negative_mass=float(neg_weight.sum(1).mean()),
            contrast_source_negative_mass=float(source_mass[eligible].mean()),
            contrast_excluded_fraction=float((neg_weight == 0).float().mean()),
            contrast_teacher_confidence=float(prob[ids].max(1).values.mean()))
    return loss, stats


class RelationalAux:
    def __init__(self, source_state, settings, device, seed, base_lr):
        self.device = torch.device(device)
        self.settings = dict(settings)
        if (not math.isfinite(settings['weight']) or settings['weight'] < 0 or
                not math.isfinite(settings['temperature']) or settings['temperature'] <= 0 or
                not 0 <= settings['mask_ratio'] < 1 or settings['block_size'] < 1 or
                settings['hidden_dim'] < 1 or settings['depth'] < 1 or
                settings['warmup_updates'] < 0 or settings['ramp_updates'] < 1):
            raise ValueError('Invalid auxiliary weight/temperature/mask/rebuilder/schedule')
        self.objective = settings['objective']
        if self.objective not in ('relation', 'probability', 'masked-ce', 'no-mask', 'latent-cosine', 'mrm-ce', 'context-relation', 'source-contrast'):
            raise ValueError('Unknown auxiliary objective')
        self.target_teacher = settings.get('target_teacher', 'ema')
        if (self.target_teacher not in ('ema', 'student') or
                (self.target_teacher != 'ema' and self.objective != 'source-contrast')):
            raise ValueError('Current-student auxiliary teacher requires source-contrast')
        self.mrm = settings.get('rebuilder', 'conv') == 'mrm'
        if settings.get('rebuilder', 'conv') not in ('conv', 'mrm'):
            raise ValueError('Unknown auxiliary rebuilder')
        if self.objective == 'mrm-ce' and not self.mrm:
            raise ValueError('MRM baseline requires the official MRM rebuilder')
        self.prepare_updates = settings.get('prepare_updates', 0)
        self.soft_kl_weight = settings.get('soft_kl_weight', 0.)
        if (type(self.prepare_updates) is not int or self.prepare_updates < 0 or
                self.prepare_updates > settings['warmup_updates']):
            raise ValueError('R preparation must be an integer within the zero-aux warmup')
        if self.prepare_updates and not self.mrm:
            raise ValueError('R preparation is supported for MRM only')
        if (not math.isfinite(self.soft_kl_weight) or self.soft_kl_weight < 0 or
                (self.soft_kl_weight and self.objective not in ('relation', 'no-mask'))):
            raise ValueError('Soft KL requires relation/no-mask and a finite nonnegative coefficient')
        threshold = settings.get('confidence_threshold', 0.)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Auxiliary confidence threshold must be in [0, 1]')
        if threshold and self.objective == 'latent-cosine':
            raise ValueError('Confidence filtering requires a probability-based auxiliary objective')
        self.view_consistency = settings.get('view_consistency', False)
        if type(self.view_consistency) is not bool:
            raise ValueError('Auxiliary view_consistency must be a boolean')
        if self.view_consistency and self.objective in ('latent-cosine', 'mrm-ce'):
            raise ValueError('Source-view filtering requires relation/probability/masked-ce/no-mask')
        if self.objective == 'context-relation':
            self.context_temperature = settings.get('context_temperature', .1)
            self.context_source_weight = settings.get('context_source_weight', 1.)
            self.encoder_grad_cap = settings.get('encoder_grad_cap', .1)
            if (not self.mrm or threshold or self.view_consistency or self.soft_kl_weight or
                    not math.isfinite(self.context_temperature) or self.context_temperature <= 0 or
                    not math.isfinite(self.context_source_weight) or self.context_source_weight < 0 or
                    not math.isfinite(self.encoder_grad_cap) or self.encoder_grad_cap < 0):
                raise ValueError('Context relation requires MRM, no class gate/KL, and finite valid metric/cap settings')
        if self.objective == 'source-contrast':
            self.encoder_grad_cap = settings.get('encoder_grad_cap', .1)
            if (not self.mrm or threshold or self.view_consistency or self.soft_kl_weight or
                    not math.isfinite(self.encoder_grad_cap) or self.encoder_grad_cap < 0 or
                    not math.isfinite(settings['contrast_temperature']) or settings['contrast_temperature'] <= 0 or
                    type(settings['contrast_grid']) is not int or settings['contrast_grid'] < 1 or
                    type(settings['contrast_keys_per_class']) is not int or settings['contrast_keys_per_class'] < 1):
                raise ValueError('Source contrast requires MRM, no class gate/KL, and valid contrast/cap settings')
        weight, bias, directions, gram = source_readout(source_state)
        self.weight = weight.to(self.device)
        self.bias = bias.to(self.device)
        self.directions = directions.to(self.device)
        self.gram = gram.to(self.device)
        self.mask_rng = torch.Generator(device=self.device).manual_seed(seed + 80017)
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 70017)
            if devices:
                torch.cuda.manual_seed(seed + 70017)
            if self.mrm:
                from core.models.mrm_rebuilder import Rebuilder
                self.rebuilder = Rebuilder(dict(type='DAFormerHead', in_channels=[64, 128, 320, 512],
                    in_index=[0, 1, 2, 3], embed_dim=settings['hidden_dim'], depth=settings['depth'],
                    mask_ratio=settings['mask_ratio'])).to(self.device)
            else:
                self.rebuilder = SpatialRebuilder(768, settings['hidden_dim'], settings['depth']).to(self.device)
            self.aux_cpu_rng = torch.get_rng_state()
            self.aux_cuda_rng = torch.cuda.get_rng_state(self.device) if devices else None
        self.optimizer = torch.optim.AdamW(self.rebuilder.parameters(), lr=base_lr * 10,
                                           weight_decay=settings['weight_decay'])
        self.last_record = None
        self.forward_count = 0

    @contextmanager
    def fork_rng(self):
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.aux_cpu_rng)
            if devices:
                torch.cuda.set_rng_state(self.aux_cuda_rng, self.device)
            try:
                yield
            finally:
                self.aux_cpu_rng = torch.get_rng_state()
                if devices:
                    self.aux_cuda_rng = torch.cuda.get_rng_state(self.device)

    def weight_at(self, iteration):
        start, ramp = self.settings['warmup_updates'], self.settings['ramp_updates']
        return self.settings['weight'] * min(1., max(0., (iteration - start) / ramp))

    def state_dict(self):
        return {'rebuilder': self.rebuilder.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'mask_rng': self.mask_rng.get_state(), 'aux_cpu_rng': self.aux_cpu_rng,
                'aux_cuda_rng': self.aux_cuda_rng, 'forward_count': self.forward_count,
                'source_weight': self.weight.cpu(), 'source_bias': self.bias.cpu(),
                'source_gram': self.gram.cpu()}

    def load_state_dict(self, state):
        for name, actual in (('source_weight', self.weight), ('source_bias', self.bias),
                             ('source_gram', self.gram)):
            if not torch.equal(state[name].to(actual.device), actual):
                raise ValueError(f'Auxiliary source anchor changed: {name}')
        self.rebuilder.load_state_dict(state['rebuilder'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.mask_rng.set_state(state['mask_rng'])
        self.aux_cpu_rng = state['aux_cpu_rng']
        self.aux_cuda_rng = state['aux_cuda_rng']
        self.forward_count = state['forward_count']

    @torch.no_grad()
    def select_teacher_sites(self, probabilities, sites, teacher, teacher_decoder,
                             full_image, geometry, full_teacher_features=None, diagnostics=False):
        full_probabilities = None
        threshold = self.settings.get('confidence_threshold', 0.)
        if self.view_consistency:
            from scripts.snd_remote import align_full
            features = teacher(full_image)[1] if full_teacher_features is None else full_teacher_features
            full_logits = F.conv2d(teacher_decoder.fuse_features(features).float(), self.weight, self.bias)
            full_logits = F.interpolate(full_logits, size=full_image.shape[-2:], mode='bilinear', align_corners=False)
            warped = align_full(full_logits, geometry, align_corners=False)
            warped = F.interpolate(warped, size=probabilities.shape[-2:], mode='bilinear', align_corners=False)
            full_probabilities = (warped / self.settings['temperature']).softmax(1)
        selected = teacher_site_mask(probabilities, sites, threshold, full_probabilities)
        record = {}
        if diagnostics:
            confidence, labels = probabilities.max(1)
            before, after = sites.squeeze(1), selected.squeeze(1)
            counts = torch.bincount(labels[before], minlength=6)
            kept = torch.bincount(labels[after], minlength=6)
            counts_list, kept_list = counts.cpu().tolist(), kept.cpu().tolist()
            record.update(aux_candidate_class_pixels=counts_list,
                aux_selected_class_pixels=kept_list,
                aux_selection_coverage_by_teacher_class=[a/b if b else None for b,a in zip(counts_list,kept_list)],
                aux_confident_fraction=float((before & (confidence >= threshold)).sum() / before.sum().clamp_min(1)))
            if full_probabilities is not None:
                disagreement = labels != full_probabilities.argmax(1)
                confident = before & (confidence >= threshold)
                record.update(aux_view_disagreement=float(disagreement[before].float().mean()) if before.any() else None,
                    aux_view_disagreement_confident=float(disagreement[confident].float().mean()) if confident.any() else None)
        return selected, record

    def backward(self, iteration, current_lr, main_loss, student, decoder, teacher,
                 teacher_decoder, full_image, unmixed_image, geometry, full_teacher_features=None):
        lam = self.weight_at(iteration)
        preparing = self.settings['weight'] > 0 and 0 < iteration <= self.prepare_updates
        self.last_record = {'aux_loss': 0., 'aux_weight': lam, 'weighted_aux_loss': 0.,
                            'mask_fraction': 0., 'aux_grad_ratio': 0., 'aux_grad_cosine': None}
        if lam == 0 and not preparing:
            return
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group['lr'] = current_lr * 10
        if self.mrm:
            return self._backward_mrm(iteration, lam, main_loss, student, decoder, teacher,
                                      teacher_decoder, full_image, unmixed_image, geometry, full_teacher_features,
                                      preparing)
        clean, pixel_valid = align_full_image(full_image, geometry, unmixed_image.shape[-2:])
        with torch.no_grad():
            teacher_latent = teacher_decoder.fuse_features(teacher(clean)[1]).detach()
        with self.fork_rng():
            student_latent = decoder.fuse_features(student(unmixed_image)[1])
        hidden = sample_block_mask(student_latent.shape[0], *student_latent.shape[-2:],
                                   self.settings['mask_ratio'], self.settings['block_size'],
                                   student_latent.device, generator=self.mask_rng)
        valid = F.adaptive_max_pool2d((~pixel_valid).float(), student_latent.shape[-2:]) == 0
        selected = hidden & valid
        candidates = selected.squeeze(1)
        predictor_mask = torch.zeros_like(hidden) if self.objective == 'no-mask' else hidden
        rebuilt = self.rebuilder(student_latent, predictor_mask)
        selection_record = {}
        relation = kl = None
        if self.objective == 'latent-cosine':
            values = 1 - F.cosine_similarity(rebuilt.float(), teacher_latent.float(), dim=1, eps=1e-6)
        else:
            temperature = self.settings['temperature']
            predicted_logits = F.conv2d(rebuilt.float(), self.weight, self.bias) / temperature
            with torch.no_grad():
                teacher_logits = F.conv2d(teacher_latent.float(), self.weight, self.bias) / temperature
                teacher_prob = teacher_logits.softmax(1)
            selected, selection_record = self.select_teacher_sites(teacher_prob, selected, teacher, teacher_decoder,
                full_image, geometry, full_teacher_features, iteration == 1 or iteration % 20 == 0)
            if self.objective == 'masked-ce':
                values = F.cross_entropy(predicted_logits, teacher_prob.argmax(1), reduction='none')
            else:
                gram = torch.eye(6, device=self.device) if self.objective == 'probability' else self.gram
                values, relation, kl = relational_with_soft_kl(predicted_logits, teacher_prob, gram,
                                                             self.soft_kl_weight)
        loss = masked_mean(values, selected.squeeze(1))
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite source-relational auxiliary loss')
        probe = (student.backbone.patch_embed1.proj.weight, decoder.linear_fuse[0].weight)
        main_grads = [p.grad.detach().clone() for p in probe] if iteration == 1 or iteration % 20 == 0 else None
        (lam * loss).backward()
        if main_grads is not None:
            ratio, cosine = gradient_alignment(main_grads, [p.grad - g for p, g in zip(probe, main_grads)])
        else:
            ratio = cosine = None
        self.forward_count += 1
        self.last_record = {'aux_loss': float(loss.detach()), 'aux_weight': lam,
                            'weighted_aux_loss': float((lam * loss).detach()),
                            'mask_fraction': float(candidates.float().sum() / valid.float().sum().clamp_min(1)),
                            'aux_grad_ratio': ratio, 'aux_grad_cosine': cosine,
                            'main_loss': float(main_loss.detach()),
                            'aux_selected_fraction': float(selected.sum() / candidates.sum().clamp_min(1)),
                            'aux_selected_cells': int(selected.sum())}
        self.last_record.update(selection_record)
        if kl is not None:
            self.last_record.update(aux_relation_loss=float(masked_mean(relation, selected.squeeze(1)).detach()),
                aux_soft_kl_loss=float(masked_mean(kl, selected.squeeze(1)).detach()),
                aux_soft_kl_weight=self.soft_kl_weight)
        if main_grads is not None and self.objective != 'latent-cosine':
            with torch.no_grad():
                confidence, labels = teacher_prob.max(1)
                keep = selected.squeeze(1)
                counts = torch.bincount(labels[keep], minlength=6)
                self.last_record.update(
                    aux_teacher_confidence=float(confidence[candidates].mean()) if candidates.any() else None,
                    aux_selected_class_fraction=(counts / counts.sum().clamp_min(1)).cpu().tolist(),
                    aux_source_main_agreement=float((labels[candidates] ==
                        teacher_decoder.linear_pred(teacher_latent).argmax(1)[candidates]).float().mean())
                        if candidates.any() else None)

    def step(self):
        if (self.last_record and (self.last_record['aux_weight'] > 0 or self.last_record.get('aux_preparing', False)) and
                self.last_record.get('aux_selected_cells', 1) > 0):
            self.optimizer.step()

    def _backward_mrm(self, iteration, lam, main_loss, student, decoder, teacher,
                      teacher_decoder, full_image, unmixed_image, geometry, full_teacher_features=None,
                      preparing=False):
        """Keep MRM's multiscale architecture; select the supervision explicitly."""
        clean, pixel_valid = align_full_image(full_image, geometry, unmixed_image.shape[-2:])
        student_target = self.target_teacher == 'student' and not preparing
        if student_target:
            teacher_latent = clean_student_target(student, decoder, clean)
        else:
            with torch.no_grad():
                teacher_features = teacher(clean)[1]
                teacher_latent = teacher_decoder.fuse_features(teacher_features)
        probe = (student.backbone.patch_embed1.proj.weight, decoder.linear_fuse[0].weight)
        main_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                      for p in probe] if not preparing and (iteration == 1 or iteration % 20 == 0) else None
        with self.fork_rng():
            with torch.no_grad() if preparing else nullcontext():
                features = student(unmixed_image)[1]
            if self.settings['block_size'] == 1:
                from core.models.mrm_rebuilder import generate_mask
                visible = generate_mask((features[-1].shape[0], 1, 16, 16), features[-1].device,
                                        self.settings['mask_ratio'], self.mask_rng)
                hidden = ~visible.bool()
            else:
                hidden = sample_block_mask(features[-1].shape[0], 16, 16,
                    self.settings['mask_ratio'], self.settings['block_size'], features[-1].device,
                    generator=self.mask_rng)
            rebuilt_features = self.rebuilder(features,
                torch.zeros_like(hidden) if self.objective == 'no-mask' else hidden)
            relation = kl = None
            if self.objective == 'mrm-ce':
                # Eq4: all valid pixels, original decoder/dropout, EMA hard pseudo-label.
                predicted_logits = decoder(rebuilt_features, unmixed_image.shape[-2:]).float()
                with torch.no_grad():
                    teacher_prob = teacher_decoder(teacher_features, unmixed_image.shape[-2:]).float().softmax(1)
                values = F.cross_entropy(predicted_logits, teacher_prob.argmax(1), reduction='none')
                valid = pixel_valid
                candidates = F.interpolate(hidden.float(), size=values.shape[-2:], mode='nearest').bool() & valid
                selected = valid
            elif self.objective == 'context-relation':
                # autograd.grad below requests encoder/R only; decoder main grads stay exact.
                rebuilt = decoder.fuse_features(rebuilt_features)
                loss, context_stats = context_relational_loss(rebuilt, teacher_latent, hidden,
                    pixel_valid, self.directions, self.gram, self.context_temperature, self.context_source_weight)
                teacher_prob = None
            elif self.objective == 'source-contrast':
                rebuilt = decoder.fuse_features(rebuilt_features)
                loss, context_stats = masked_source_contrastive_loss(rebuilt, teacher_latent, hidden,
                    pixel_valid, self.weight, self.bias, self.gram, self.settings['contrast_temperature'],
                    self.settings['temperature'], self.settings['contrast_grid'],
                    self.settings['contrast_keys_per_class'], self.mask_rng)
                context_stats.update(aux_target_teacher='student' if student_target else 'ema',
                                     aux_target_teacher_requested=self.target_teacher)
                if student_target and (iteration == 1 or iteration % 20 == 0):
                    with torch.no_grad():
                        ema_latent = teacher_decoder.fuse_features(teacher(clean)[1])
                        size = self.settings['contrast_grid']
                        current = F.adaptive_avg_pool2d(teacher_latent.float(), size)
                        history = F.adaptive_avg_pool2d(ema_latent.float(), size)
                        valid = F.adaptive_max_pool2d((~pixel_valid).float(), size).squeeze(1) == 0
                        current_prob = F.adaptive_avg_pool2d((F.conv2d(teacher_latent.float(), self.weight, self.bias) / self.settings['temperature']).softmax(1), size)
                        history_prob = F.adaptive_avg_pool2d((F.conv2d(ema_latent.float(), self.weight, self.bias) / self.settings['temperature']).softmax(1), size)
                        context_stats.update(
                            aux_target_ema_cosine=float(F.cosine_similarity(current, history, dim=1)[valid].mean()) if valid.any() else None,
                            aux_target_ema_source_agreement=float((current_prob.argmax(1) == history_prob.argmax(1))[valid].float().mean()) if valid.any() else None,
                            aux_target_ema_probability_l1=float((current_prob-history_prob).abs().sum(1)[valid].mean()) if valid.any() else None)
                teacher_prob = None
            else:
                rebuilt = decoder.fuse_features(rebuilt_features)
                valid = F.adaptive_max_pool2d((~pixel_valid).float(), rebuilt.shape[-2:]) == 0
                candidates = F.interpolate(hidden.float(), size=rebuilt.shape[-2:], mode='nearest').bool() & valid
                selected = candidates
                if self.objective == 'latent-cosine':
                    teacher_prob = None
                    values = 1 - F.cosine_similarity(rebuilt.float(), teacher_latent.float(), dim=1, eps=1e-6)
                else:
                    temperature = self.settings['temperature']
                    predicted_logits = F.conv2d(rebuilt.float(), self.weight, self.bias) / temperature
                    with torch.no_grad():
                        teacher_prob = (F.conv2d(teacher_latent.float(), self.weight, self.bias) / temperature).softmax(1)
                    if self.objective == 'masked-ce':
                        values = F.cross_entropy(predicted_logits, teacher_prob.argmax(1), reduction='none')
                    else:
                        gram = torch.eye(6, device=self.device) if self.objective == 'probability' else self.gram
                        values, relation, kl = relational_with_soft_kl(predicted_logits, teacher_prob, gram,
                                                                     self.soft_kl_weight)
        if self.objective in ('context-relation', 'source-contrast'):
            return self._backward_context(loss, context_stats, lam, main_loss, student, preparing)
        selection_record = {}
        if teacher_prob is not None:
            selected, selection_record = self.select_teacher_sites(teacher_prob, selected, teacher, teacher_decoder,
                full_image, geometry, full_teacher_features,
                main_grads is not None or (preparing and (iteration == 1 or iteration % 20 == 0)))
        loss = masked_mean(values, selected.squeeze(1))
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite MRM auxiliary loss')
        if preparing:
            # Only R receives gradients; existing main gradients on encoder/decoder stay exact.
            torch.autograd.backward(loss, inputs=tuple(p for p in self.rebuilder.parameters() if p.requires_grad))
        else:
            (lam * loss).backward()
        ratio, cosine = (gradient_alignment(main_grads,
            [p.grad - g for p, g in zip(probe, main_grads)]) if main_grads is not None else (None, None))
        if preparing:
            ratio = 0.
        self.forward_count += 1
        self.last_record = dict(aux_loss=float(loss.detach()), aux_weight=lam,
            weighted_aux_loss=float((lam * loss).detach()), main_loss=float(main_loss.detach()),
            mask_fraction=float(candidates.sum() / valid.sum().clamp_min(1)),
            aux_selected_fraction=float(selected.sum() / valid.sum().clamp_min(1)) if self.objective == 'mrm-ce'
                else float(selected.sum() / candidates.sum().clamp_min(1)),
            aux_selected_cells=int(selected.sum()), aux_grad_ratio=ratio, aux_grad_cosine=cosine)
        self.last_record.update(selection_record)
        if self.prepare_updates:
            self.last_record.update(aux_preparing=preparing, rebuilder_update_weight=1. if preparing else lam)
        if kl is not None:
            self.last_record.update(aux_relation_loss=float(masked_mean(relation, selected.squeeze(1)).detach()),
                aux_soft_kl_loss=float(masked_mean(kl, selected.squeeze(1)).detach()),
                aux_soft_kl_weight=self.soft_kl_weight)
        if teacher_prob is not None and (main_grads is not None or
                                        (preparing and (iteration == 1 or iteration % 20 == 0))):
            with torch.no_grad():
                keep = selected.squeeze(1)
                confidence, labels = teacher_prob.max(1)
                counts = torch.bincount(labels[keep], minlength=6)
                self.last_record.update(aux_teacher_confidence=float(confidence[keep].mean()) if keep.any() else None,
                    aux_selected_class_fraction=(counts / counts.sum().clamp_min(1)).cpu().tolist())
                source_logits = F.conv2d(teacher_latent.float(), self.weight, self.bias)
                main_logits = teacher_decoder.linear_pred(teacher_latent)
                if source_logits.shape[-2:] != labels.shape[-2:]:
                    source_logits = F.interpolate(source_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                    main_logits = F.interpolate(main_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                sites = candidates.squeeze(1)
                self.last_record['aux_source_main_agreement'] = float((source_logits.argmax(1)[sites] ==
                    main_logits.argmax(1)[sites]).float().mean()) if sites.any() else None

    def _backward_context(self, loss, stats, lam, main_loss, student, preparing):
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite context relational loss')
        encoder = tuple(p for p in student.parameters() if p.requires_grad) if not preparing else ()
        rebuilder = tuple(p for p in self.rebuilder.parameters() if p.requires_grad)
        gradients = torch.autograd.grad(loss, encoder + rebuilder, allow_unused=True)
        egrads, rgrads = gradients[:len(encoder)], gradients[len(encoder):]
        main_sq = loss.detach().new_zeros(())
        aux_sq, dot = main_sq.clone(), main_sq.clone()
        for p, g in zip(encoder, egrads):
            if p.grad is not None:
                main_sq += p.grad.detach().float().square().sum()
            if g is not None:
                aux_sq += g.detach().float().square().sum()
                if p.grad is not None:
                    dot += (p.grad.detach().float()*g.detach().float()).sum()
        main_norm, aux_norm = main_sq.sqrt(), aux_sq.sqrt()
        r_norm = sum((g.detach().float().square().sum() for g in rgrads if g is not None),
                     loss.detach().new_zeros(())).sqrt()
        if not torch.isfinite(main_norm + aux_norm + r_norm + dot):
            raise FloatingPointError('Non-finite context encoder gradients')
        effective = min(lam, float(self.encoder_grad_cap*main_norm/aux_norm.clamp_min(1e-12))) if encoder else 0.
        cosine = float((dot/(main_norm*aux_norm)).clamp(-1, 1)) if main_norm*aux_norm > 1e-12 and effective else None
        for p, g in zip(encoder, egrads):
            if g is not None and effective:
                if p.grad is None:
                    p.grad = g.detach()*effective
                else:
                    p.grad.add_(g.detach(), alpha=effective)
        for p, g in zip(rebuilder, rgrads):
            p.grad = g.detach() if g is not None else None
        self.forward_count += 1
        self.last_record = dict(stats, aux_loss=float(loss.detach()), aux_weight=lam,
            weighted_aux_loss=effective*float(loss.detach()), main_loss=float(main_loss.detach()),
            aux_preparing=preparing, rebuilder_update_weight=1., aux_encoder_weight=effective,
            aux_encoder_cap_active=effective < lam, aux_encoder_grad_cap=self.encoder_grad_cap,
            aux_grad_ratio=float(effective*aux_norm/main_norm.clamp_min(1e-12)), aux_grad_cosine=cosine,
            aux_main_encoder_grad_norm=float(main_norm), aux_encoder_grad_norm=float(aux_norm),
            aux_rebuilder_grad_norm=float(r_norm),
            aux_gradient_scope='all_encoder', aux_decoder_gradient_weight=0.)
