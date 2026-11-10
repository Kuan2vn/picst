"""Exploratory support routing and frozen-source gates; target train GT is never read."""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scripts.gsd_reliability import GSDSupervision, class_topq, normalized_js, warp_to_native


def overlap_matrix(target_cells, source_cells, target_start, target_span, source_start, source_span, device):
    a = torch.arange(target_cells+1, device=device, dtype=torch.float32)*target_span/target_cells+target_start
    b = torch.arange(source_cells+1, device=device, dtype=torch.float32)*source_span/source_cells+source_start
    return (torch.minimum(a[1:, None], b[None, 1:])-torch.maximum(a[:-1, None], b[None, :-1])).clamp_min(0)/(target_span/target_cells)


def project_probabilities(native, geometry, cells):
    pooled, coverage = [], []
    for index, g in enumerate(geometry):
        wy = overlap_matrix(cells[0], native.shape[-2], g['view_y'], g['view_size'], g['native_y'], g['native_size'], native.device)
        wx = overlap_matrix(cells[1], native.shape[-1], g['view_x'], g['view_size'], g['native_x'], g['native_size'], native.device)
        support = wy.sum(1)[:, None]*wx.sum(1)[None, :]
        value = torch.matmul(torch.matmul(wy, native[index]), wx.T)
        pooled.append(value/support.clamp_min(1e-8))
        coverage.append(support >= 1-1e-5)
    return torch.stack(pooled), torch.stack(coverage)


def macro_regions(labels, valid, gsd, minimum_area):
    """Area is measured BEFORE weak resize or donor enlargement; ignore truncated components."""
    result = np.zeros(labels.shape, dtype=bool)
    for index, (image, support) in enumerate(zip(labels, valid)):
        interior = cv2.erode(support.astype(np.uint8), np.ones((3,3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
        for cls in range(6):
            count, components, stats, _ = cv2.connectedComponentsWithStats(((image == cls)&support).astype(np.uint8), connectivity=8)
            touching = np.unique(components[interior == 0])
            keep = [j for j in range(1,count) if j not in touching and stats[j,cv2.CC_STAT_AREA]*gsd**2 > minimum_area]
            result[index] |= np.isin(components, keep)
    return result


def contradiction(source, labels, confidence=.9, class_probability=.05):
    legal = labels != 255
    assigned = source.gather(1, labels.clamp(0, source.shape[1]-1)[:,None]).squeeze(1)
    return legal & (source.max(1).values >= confidence) & (source.argmax(1) != labels) & (assigned <= class_probability)


def filter_bank_components(labels, veto, fraction):
    result = labels.copy()
    for cls in range(6):
        count, components = cv2.connectedComponents((labels == cls).astype(np.uint8), connectivity=8)
        total = np.bincount(components.ravel(), minlength=count)
        rejected = np.bincount(components.ravel(), weights=veto.ravel(), minlength=count)
        bad = np.flatnonzero((rejected/np.maximum(total,1) >= fraction)&(np.arange(count)>0))
        result[np.isin(components,bad)] = 255
    return result


class ScoutSupervision(GSDSupervision):
    def __init__(self, runtime, config):
        super().__init__(runtime,config)
        self.scout = config['scout']
        required = ('variant','support','bank_veto','veto_floor','fusion_gamma','minimum_area_m2')
        if any(k not in self.scout for k in required):
            raise ValueError('Incomplete scout configuration')
        if not (0 <= self.scout['veto_floor'] <= 1 and 0 <= self.scout['fusion_gamma'] <= 1 and self.scout['minimum_area_m2'] > 0):
            raise ValueError('Invalid scout thresholds')
        self.source_models = None
        self.cache = Path(runtime.scout_source_cache)/runtime.source_sha
        self.cache.mkdir(parents=True,exist_ok=True)
        self.cache_forwards = 0
        self.bank_removed = np.zeros(6,dtype=np.int64)

    @torch.no_grad()
    def source_probability(self, name):
        path = self.cache/(name+'.npy')
        if path.exists():
            logits = torch.from_numpy(np.load(path,allow_pickle=False).copy()).to(self.runtime.device).float()[None]
        else:
            if self.source_models is None:
                from core.models.segformer import build_segformer_b2
                from train_SRMP_remote import load_checkpoint, rng_state, restore_rng
                rng = rng_state()
                try:
                    encoder, classifier = build_segformer_b2(6)
                    state = load_checkpoint(self.runtime.source_checkpoint)
                    encoder.load_state_dict(state['feature_extractor'],strict=True)
                    classifier.load_state_dict(state['classifier'],strict=True)
                    self.source_models = encoder.to(self.runtime.device).eval(), classifier.to(self.runtime.device).eval()
                finally:
                    restore_rng(rng)
            from scripts.snd_remote import normalize
            encoder,classifier = self.source_models
            logits = classifier(encoder(normalize(self.runtime.image(name))[None].to(self.runtime.device))[1])
            # Always use the same fp16 cached logits, including the first use.
            values = logits[0].cpu().numpy().astype(np.float16)
            temporary = path.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:np.save(stream,values,allow_pickle=False)
            temporary.replace(path)
            logits = torch.from_numpy(values).to(self.runtime.device).float()[None]
            self.cache_forwards += 1
        if tuple(logits.shape) != (1,6,128,128) or not torch.isfinite(logits).all():
            raise ValueError('Invalid frozen source cache')
        return F.interpolate(logits,size=(512,512),mode='bilinear',align_corners=False).softmax(1)

    @torch.no_grad()
    def bank_filter(self, name, selected):
        if not self.scout['bank_veto']:return selected
        full = cv2.resize(selected,(512,512),interpolation=cv2.INTER_NEAREST)
        labels = torch.from_numpy(full.astype(np.int64))[None].to(self.runtime.device)
        veto = contradiction(self.source_probability(name),labels,self.scout.get('source_confidence',.9),
                             self.scout.get('source_class_probability',.05)).cpu().numpy()[0]
        filtered = filter_bank_components(full,veto,self.scout.get('bank_veto_fraction',.25))
        before = np.bincount(full[full!=255],minlength=6)
        after = np.bincount(filtered[filtered!=255],minlength=6)
        self.bank_removed += before-after
        return cv2.resize(filtered,(selected.shape[1],selected.shape[0]),interpolation=cv2.INTER_NEAREST)

    @torch.no_grad()
    def prepare(self, names, parameters, native_probability, encoder, classifier, native_full_logits=None):
        if native_full_logits is None:raise ValueError('Scout needs the native teacher logits')
        # The decoder captures its true pre-upsample logits; no extra native forward.
        native_decoder = classifier.scout_native_logits
        self.native_probability = native_probability.detach()
        score_confidence = native_probability.max(1).values
        self.fused_label = native_probability.argmax(1)
        self.macro = torch.zeros_like(self.fused_label,dtype=torch.bool)
        self.veto = torch.zeros_like(self.macro)
        self.last_record = {'scout_variant':self.scout['variant']}
        if self.scout['support']:
            view = self.config['generic_view'] if self.config['arm']=='generic' else self.config['gsd_view']
            images,support,geometry = self._load(names,view)
            pooled,common = project_probabilities(native_decoder.softmax(1),geometry,native_decoder.shape[-2:])
            coarse_valid = F.interpolate(support[:,None].float(),size=native_decoder.shape[-2:],mode='area')[:,0]>=1-1e-5
            common &= coarse_valid
            homogeneous = pooled.max(1).values>=self.scout.get('homogeneous_probability',.8)
            valid_coarse = common
            aligned_pool,_ = warp_to_native(pooled,geometry,valid_coarse,normalize=False)
            valid_map,valid_geometry = warp_to_native(valid_coarse[:,None].float(),geometry,valid_coarse,normalize=False)
            valid_full = valid_geometry & (valid_map[:,0]>=1-1e-5)
            homogeneous_map,_ = warp_to_native(homogeneous[:,None].float(),geometry,valid_coarse,normalize=False)
            full_prob = native_full_logits.softmax(1)
            macro = macro_regions(full_prob.argmax(1).cpu().numpy(),valid_full.cpu().numpy(),
                                  self.config['target_gsd_m'],self.scout['minimum_area_m2'])
            macro = torch.from_numpy(macro).to(self.runtime.device)
            self.macro = self.runtime.full2weak(macro[:,None].float(),parameters)[:,0]>=1-1e-5
            self.last_valid = self.runtime.full2weak(valid_full[:,None].float(),parameters)[:,0]>=1-1e-5
            pooled_weak = self.runtime.full2weak(aligned_pool,parameters)
            homogeneous_weak = self.runtime.full2weak(homogeneous_map,parameters)[:,0]>=1-1e-5
            self.macro &= self.last_valid & homogeneous_weak & (pooled_weak.argmax(1)==self.fused_label)
            divergence = torch.zeros_like(self.fused_label,dtype=torch.float)
            if self.config['arm']!='confidence':
                modes=encoder.training,classifier.training
                encoder.eval();classifier.eval()
                try:
                    coarse_logits=classifier(encoder(images)[1])
                finally:encoder.train(modes[0]);classifier.train(modes[1])
                coarse=coarse_logits.softmax(1)
                distance=normalized_js(pooled,coarse)
                aligned,_=warp_to_native(distance[:,None],geometry,valid_coarse,normalize=False)
                divergence=self.runtime.full2weak(aligned,parameters)[:,0]
                self.extra_forwards+=len(names)
                if self.scout['fusion_gamma']:
                    aligned_coarse,_=warp_to_native(coarse_logits,geometry,valid_coarse,normalize=False)
                    coarse_weak=self.runtime.full2weak(aligned_coarse,parameters).softmax(1)
                    gamma=self.scout['fusion_gamma']
                    fusion=(1-gamma)*native_probability+gamma*coarse_weak
                    active=self.macro & (coarse_weak.max(1).values>=.9)
                    self.fused_label=torch.where(active,fusion.argmax(1),self.fused_label)
                    score_confidence=torch.where(active,fusion.max(1).values,score_confidence)
            self.last_score=score_confidence*torch.exp(-self.config['lambda']*divergence)
            self.last_record.update(gsd_js_mean=float(divergence[self.macro].mean()) if self.macro.any() else None,
                                    scout_macro_fraction=float(self.macro.float().mean()))
        else:
            self.last_valid=parameters['valid'].to(self.runtime.device).bool()
            self.last_score=native_probability.max(1).values
        if self.scout['veto_floor']<1:
            source=torch.cat([self.source_probability(name) for name in names])
            source=self.runtime.full2weak(source,parameters)
            source=source/source.sum(1,keepdim=True).clamp_min(1e-8)
            self.veto=contradiction(source,self.fused_label,self.scout.get('source_confidence',.9),
                                    self.scout.get('source_class_probability',.05))
        self.last_record.update(scout_source_cache_inferences=self.cache_forwards,
                                scout_bank_removed_counts=self.bank_removed.tolist(),
                                gsd_extra_teacher_images=self.extra_forwards)

    def labels_for_loss(self, labels, pasted):
        if not self.scout['fusion_gamma']:return labels
        return torch.where((labels!=255)&(pasted==255),self.fused_label,labels)

    def loss(self, loss_map, weights, labels, pasted):
        recipient=(labels!=255)&(pasted==255)
        coefficient=torch.ones_like(loss_map)
        selected=torch.zeros_like(recipient)
        eligible=recipient & self.macro
        if self.scout['support']:
            selected=class_topq(self.last_score,labels,eligible,self.config['coverage'])
            scale=eligible.sum()/selected.sum().clamp_min(1)
            alpha=self.config.get('intervention',1.)
            coefficient=torch.where(eligible,1-alpha+alpha*scale*selected,coefficient)
        coefficient=torch.where(recipient & self.veto,self.scout['veto_floor']*coefficient,coefficient)
        weighted=loss_map*weights
        origins={'donor':(labels!=255)&(pasted!=255), 'recipient_support':recipient&self.last_valid,
                 'recipient_outside':recipient&~self.last_valid}
        self.last_record.update(gsd_loss_mode=self.config['loss_mode'],
            scout_veto_fraction=float((recipient&self.veto).float().mean()),
            gsd_recipient_coefficient_mean=float(coefficient[recipient].mean()) if recipient.any() else None,
            gsd_recipient_eligible_counts=torch.bincount(labels[eligible],minlength=6).cpu().tolist(),
            gsd_recipient_selected_counts=torch.bincount(labels[selected],minlength=6).cpu().tolist(),
            scout_origin_ce={k:float(weighted[m].sum().detach()) for k,m in origins.items()},
            scout_origin_weighted_ce={k:float((weighted*coefficient)[m].sum().detach()) for k,m in origins.items()})
        return (weighted*coefficient).mean()
