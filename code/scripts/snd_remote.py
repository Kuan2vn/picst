"""Manifest/data adapter for the existing SND trainer (no target GT access).

SND equations and bi-level optimization live in train_SND.py. This module
adapts author candidate ranking, balanced sampling and object pasting to
square remote-sensing patches. See REPRODUCE_SND_VH2PD.md for deviations.
"""
from __future__ import annotations

import json
import math
import logging
import pickle
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label as connected_components
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from core.datasets.transform import ColorJitter, GaussianBlur
from train_SRMP_remote import ManifestDataset, load_checkpoint, rng_state, restore_rng


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def normalize(image):
    return TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def align_full(feat, params, down_ratio=1, nearest=False, align_corners=True):
    """Scale, symmetric pad, crop and flip in the same order as the image.

For encoder features preserve the author's /8 coordinate convention.
Padding is tracked explicitly; padded pseudo-label pixels are ignored.
"""
    outputs = []
    for i in range(len(feat)):
        values = {k: [int(v[i]) for v in params[k]] for k in
                  ('rescale_size', 'padding', 'random_crop_axis')}
        h, w = values['rescale_size']
        kwargs = {} if nearest else {'mode': 'bilinear', 'align_corners': align_corners}
        x = F.interpolate(feat[i:i+1], size=(max(1, h//down_ratio), max(1, w//down_ratio)), **kwargs)
        left, top, right, bottom = values['padding']
        x = F.pad(x, (left//down_ratio, right//down_ratio, top//down_ratio, bottom//down_ratio))
        y1, y2, x1, x2 = values['random_crop_axis']
        th, tw = (y2-y1)//down_ratio, (x2-x1)//down_ratio
        x = x[:, :, y1//down_ratio:y1//down_ratio+th, x1//down_ratio:x1//down_ratio+tw]
        # Rounding of odd padding/feature coordinates can remove one border cell.
        x = F.pad(x, (0, max(0, tw-x.shape[-1]), 0, max(0, th-x.shape[-2])))
        if params['RandomHorizontalFlip'][i]:
            x = x.flip(-1)
        outputs.append(x)
    return torch.cat(outputs)


class ObjectMixer:
    """Author component-paste geometry; dataset paths/classes are generic.

    The 50-pixel eligibility check is done once, preventing the unbounded
    rejection loop in the original mixer when a rare class has no object.
    """
    def __init__(self, runtime, bank='CTR', meta=False):
        self.runtime, self.bank, self.meta = runtime, bank, meta
        self.label_to_file = {c: [] for c in range(6)}
        for path in sorted((runtime.output / bank).glob('*.png')):
            counts = np.bincount(np.asarray(Image.open(path)).ravel(), minlength=256)
            for c in range(6):
                if counts[c] >= 50:
                    self.label_to_file[c].append(path.name)

    def mix(self, image, classes, probabilities, recipient_scale=None, return_stats=False):
        output = np.array(image, copy=True)
        labels = np.full(output.shape[:2], 255, dtype=np.int64)
        settings = getattr(self.runtime, 'geometry_settings', None)
        if return_stats:
            from scripts.dtst_geometry import empty_geometry_stats
            stats = empty_geometry_stats()
        valid = [j for j, c in enumerate(classes) if self.label_to_file.get(c)]
        if not valid:
            return (output, labels, stats) if return_stats else (output, labels)
        p = np.asarray(probabilities, dtype=float)[valid]
        p = p / p.sum() if p.sum() > 0 else np.ones(len(valid))/len(valid)
        cls = int(classes[int(np.random.choice(valid, p=p))])
        name = random.choice(self.label_to_file[cls])
        donor = np.array(self.runtime.image(name))
        mask = np.asarray(Image.open(self.runtime.output/self.bank/name))
        if mask.shape != donor.shape[:2]:
            raise ValueError('Candidate/image geometry mismatch')
        components, count = connected_components(mask == cls, connectivity=2, return_num=True)
        occupied = np.zeros(output.shape[:2], dtype=bool)
        height, width = output.shape[:2]
        for component in range(1, count+1):
            instance = (components == component).astype(np.uint8)
            if instance.sum() < 20:
                continue
            contours, _ = cv2.findContours(instance, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            draw = np.random.rand()
            if settings is not None and not self.meta:
                from scripts.dtst_geometry import donor_scale
                scale = donor_scale(settings['donor_mode'], recipient_scale, draw)
            else:
                scale = 1.0 + draw * (0.2 if self.meta else 1.5)
            if return_stats:
                stats['proposed'][cls] += 1
            crop = cv2.resize(donor[y:y+h, x:x+w], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            crop_mask = cv2.resize(instance[y:y+h, x:x+w], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
            ch, cw = crop.shape[:2]
            if ch >= height or cw >= width:
                if return_stats:
                    stats['skipped_size'][cls] += 1
                continue
            for _ in range(99):
                oy, ox = random.randint(0, height-ch-1), random.randint(0, width-cw-1)
                if not occupied[oy, ox] and not occupied[oy+ch, ox+cw]:
                    output[oy:oy+ch, ox:ox+cw][crop_mask] = crop[crop_mask]
                    labels[oy:oy+ch, ox:ox+cw][crop_mask] = cls
                    occupied[oy:oy+ch, ox:ox+cw] = True
                    if return_stats:
                        stats['accepted'][cls] += 1
                        stats['paste_pixels'][cls] += crop_mask.sum()
                        # Use rounded component dimensions, not only the proposed scale.
                        stats['pitch_ratio_sum'][cls] += recipient_scale/math.sqrt((ch/h)*(cw/w))
                        stats['resize_sum'][cls] += math.sqrt((ch/h)*(cw/w))
                    break
        return (output, labels, stats) if return_stats else (output, labels)


class TargetTraining(Dataset):
    def __init__(self, runtime, cfg):
        self.runtime, self.cfg = runtime, cfg
        self.mix_classes = list(range(6))  # No street-scene class IDs/co-occurrence rules.
        # Workers must see the trainer's confidence-based probability updates.
        self._mix_p = torch.full((6,), 1/6, dtype=torch.float64).share_memory_()
        self.mixer = ObjectMixer(runtime)
        self.blur = GaussianBlur()
        self.jitter = ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.2)
        with (runtime.output/'CTR_O.p').open('rb') as handle:
            label_to_file, file_to_label = pickle.load(handle)
        available = [c for c in range(6) if label_to_file[c]]
        if not available:
            raise RuntimeError('SND candidate bank has no valid classes')
        # Author inverse-log image sampling, sub-epochs of 500. Restrict choices
        # to nonempty classes instead of dropping singleton classes/samples.
        self.names = []
        indices = {c: 0 for c in available}
        for start in range(0, cfg.SOLVER.MAX_ITER*cfg.SOLVER.BATCH_SIZE, 500):
            distribution = np.zeros(6)
            for _ in range(min(500, cfg.SOLVER.MAX_ITER*cfg.SOLVER.BATCH_SIZE-start)):
                fraction = distribution / max(distribution.sum(), 1)
                p = 1 / np.log(1.01 + fraction[available])
                c = int(np.random.choice(available, p=p/p.sum()))
                files = label_to_file[c]
                if indices[c] >= len(files):
                    random.shuffle(files)
                    indices[c] = 0
                name = files[indices[c]]
                indices[c] += 1
                self.names.append(name)
                distribution[file_to_label[name]] += 1

    def __len__(self):
        return len(self.names)

    @property
    def mix_p(self):
        return self._mix_p.numpy().copy()

    @mix_p.setter
    def mix_p(self, value):
        self._mix_p.copy_(torch.as_tensor(value, dtype=torch.float64))

    def __getitem__(self, index):
        name = self.names[index]
        full = self.runtime.image(name)
        image, _, params = self.blur(full, None, {})
        image, _, params = self.jitter(image, None, params)
        crop = self.runtime.size
        scale = random.uniform(1., 2.) if self.runtime.architecture == 'segformer_mit_b2_nvidia_mlp768' else random.uniform(.75, 1.25)
        size = int(crop * scale)
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        pad = max(0, crop-size)
        image = TF.pad(image, [pad, pad, pad, pad])
        valid = TF.pad(Image.new('L', (size, size), 1), [pad, pad, pad, pad])
        y, x = random.randint(0, image.height-crop), random.randint(0, image.width-crop)
        box = (x, y, x+crop, y+crop)
        image, valid = image.crop(box), valid.crop(box)
        flip = random.random() < .5
        if flip:
            image, valid = TF.hflip(image), TF.hflip(valid)
        params = {'rescale_size': (size, size), 'padding': (pad, pad, pad, pad),
                  'random_crop_axis': (y, y+crop, x, x+crop),
                  'RandomHorizontalFlip': flip, 'valid': torch.from_numpy(np.array(valid, dtype=bool))}
        if getattr(self.runtime, 'geometry_settings', None) is not None:
            mixed, mix_label, stats = self.mixer.mix(image, self.mix_classes, self.mix_p,
                                                    recipient_scale=size/crop, return_stats=True)
            params['geometry_stats'] = stats
            params['recipient_scale'] = size/crop
        else:
            mixed, mix_label = self.mixer.mix(image, self.mix_classes, self.mix_p)
        result = (normalize(mixed), torch.full((crop, crop), 255), name, params,
                  normalize(full), torch.from_numpy(mix_label))
        extra = (getattr(self.runtime, 'aux', None) is not None or getattr(self.runtime, 'context_settings', None) is not None
                 or getattr(getattr(self.runtime, 'baseline', None), 'needs_unmixed', False))
        return result + (normalize(image),) if extra else result


class SNDRemoteRuntime:
    def __init__(self, root, output, device, size=512, limit=0, workers=2, signature='',
                 architecture='resnet101_aspp'):
        self.root, self.output, self.device = Path(root), Path(output), torch.device(device)
        self.size, self.workers, self.signature = size, workers, signature
        self.architecture = architecture
        self.target_split = 'target_train' if architecture == 'segformer_mit_b2_nvidia_mlp768' else 'isprs_pd_rgb_train'
        self.eval_split = 'target_eval' if architecture == 'segformer_mit_b2_nvidia_mlp768' else 'isprs_pd_rgb_eval'
        dataset = ManifestDataset(self.root, self.target_split, 'target', 6, size)
        rows = dataset.rows
        if limit:
            rows = [rows[i] for i in np.linspace(0, len(rows)-1, min(limit, len(rows)), dtype=int)]
        self.images = {p.name: p for p, _ in rows}
        if len(self.images) != len(rows):
            raise ValueError('Target filenames must be unique')
        self.last_iteration = 0
        self.meta_mixer = None
        self.eval_period, self.eval_limit, self.stop_iteration = 0, 0, None
        self.evaluated_iterations = set()
        self.meta_images = 0
        self.teacher_inferences = 0
        self.bank_refreshes = 0

    def image(self, name):
        # All paths come exclusively from the unlabeled target-train manifest.
        with Image.open(self.images[name]) as im:
            im = im.convert('RGB')
            return im.resize((self.size, self.size), Image.Resampling.BICUBIC) if im.size != (self.size, self.size) else im.copy()

    def build_feature_extractor(self, cfg):
        if self.architecture == 'segformer_mit_b2_nvidia_mlp768':
            from core.models.segformer import build_segformer_b2
            return build_segformer_b2(6)[0]
        from core.models.feature_extractor import resnet_feature_extractor
        return resnet_feature_extractor('resnet101', aux=True, pretrained_backbone=False, freeze_bn=True)

    def build_classifier(self, cfg):
        if self.architecture == 'segformer_mit_b2_nvidia_mlp768':
            from core.models.segformer import SegFormerB2Decoder
            return SegFormerB2Decoder(6)
        from core.models import build_classifier
        return build_classifier(cfg)

    def load_initial(self, path):
        state = load_checkpoint(Path(path))
        if self.architecture == 'segformer_mit_b2_nvidia_mlp768':
            from train_SRMP_remote import B2_PROTOCOL, sha256_file
            if (state.get('architecture') != self.architecture or state.get('protocol') != B2_PROTOCOL or
                    state.get('smoke_only', False) != getattr(self, 'smoke_only', False) or
                    state.get('source_checkpoint_sha256', getattr(self, 'source_sha', None)) != getattr(self, 'source_sha', None)):
                raise ValueError('B2 source/warm checkpoint provenance mismatch')
            if state.get('stage') == 'warm' and state.get('protocol_signature') != getattr(self, 'warm_signature', self.signature):
                raise ValueError('Warm checkpoint signature mismatch')
            if state.get('stage') not in ('source', 'warm'):
                raise ValueError('B2 initialization must be a source or warm checkpoint')
            return state
        return self.load_source(path)

    @staticmethod
    def load_source(path):
        state = load_checkpoint(Path(path))
        if state.get('architecture') == 'segformer_mit_b2_nvidia_mlp768':
            if (state.get('task') != 'vh2pd' or state.get('stage') != 'source' or
                    state.get('source_split') != 'source_train' or state.get('num_classes') != 6 or
                    state.get('checkpoint_selection') not in
                    ('target_validation_selected', 'source_validation_selected', 'last_scheduled')):
                raise ValueError('Invalid B2 source checkpoint')
            if not 0 < state.get('iteration', 0) <= state.get('training_settings', {}).get('steps', 0):
                raise ValueError('Invalid B2 source training budget')
            return state
        expected = {'task': 'vh2pd', 'stage': 'source', 'num_classes': 6,
                    'source_split': 'isprs_vh_irrg_train', 'normalization': 'imagenet_rgb_planes'}
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f'Source checkpoint mismatch: {key}')
        # Initialization compatibility does not depend on the checkpoint-selection rule.
        iteration = state.get('iteration')
        budget = state.get('training_settings', {}).get('steps', iteration)
        if (type(iteration) is not int or type(budget) is not int
                or not 0 < iteration <= budget):
            raise ValueError('Source checkpoint has an invalid training iteration/budget')
        return state

    def build_dataset(self, cfg, mode='train', is_source=False, rsc=None):
        if mode != 'train' or is_source:
            raise ValueError('Adaptation adapter only exposes unlabeled target training')
        return TargetTraining(self, cfg)

    def full2weak(self, feat, params, down_ratio=1, nearest=False):
        return align_full(feat, params, down_ratio, nearest,
                          align_corners=self.architecture != 'segformer_mit_b2_nvidia_mlp768')

    def align_features(self, features, params):
        """Align four scales using pixel coordinates, including odd crop offsets."""
        if len(features) != 4:
            raise ValueError('Expected four B2 feature scales')
        aligned = []
        for stride, feature in zip((4, 8, 16, 32), features):
            output_height = math.ceil(self.size / stride)
            output_width = math.ceil(self.size / stride)
            yy, xx = torch.meshgrid(torch.arange(output_height, device=feature.device),
                                    torch.arange(output_width, device=feature.device), indexing='ij')
            rows = []
            for index in range(feature.shape[0]):
                scaled_height = int(params['rescale_size'][0][index])
                scaled_width = int(params['rescale_size'][1][index])
                left = int(params['padding'][0][index]); top = int(params['padding'][1][index])
                y1 = int(params['random_crop_axis'][0][index]); x1 = int(params['random_crop_axis'][2][index])
                flip = bool(params['RandomHorizontalFlip'][index])
                crop_x = self.size - 1 - xx * stride if flip else xx * stride
                source_x = ((crop_x + x1 - left + .5) * self.size / scaled_width - .5) / stride
                source_y = ((yy * stride + y1 - top + .5) * self.size / scaled_height - .5) / stride
                grid = torch.stack((2 * (source_x + .5) / feature.shape[-1] - 1,
                                    2 * (source_y + .5) / feature.shape[-2] - 1), -1).unsqueeze(0)
                rows.append(F.grid_sample(feature[index:index+1], grid, mode='bilinear',
                                          padding_mode='zeros', align_corners=False))
            aligned.append(torch.cat(rows, 0))
        return tuple(aligned)

    @staticmethod
    def mask_padding(pseudo, params):
        return pseudo.masked_fill(~params['valid'].to(pseudo.device), 255)

    @torch.no_grad()
    def run_candidate(self, cfg, encoder, classifier, local_rank, distributed, init_candidate=False, update_meta=False):
        if distributed:
            raise ValueError('SND remote runner is single-GPU')
        modes = encoder.training, classifier.training
        encoder.eval(); classifier.eval()
        names = list(self.images)
        self.teacher_inferences += len(names)
        self.bank_refreshes += 1
        half = self.size // 2
        labels = np.empty((len(names), half, half), dtype=np.uint8)
        confidence = np.empty(labels.shape, dtype=np.float32) if init_candidate else None
        stability = np.zeros((len(names), 6))
        for index, name in enumerate(names):
            probs = classifier(encoder(normalize(self.image(name)).unsqueeze(0).to(self.device))[1], (self.size, self.size)).softmax(1)
            prob, pred = probs.max(1)
            labels[index] = F.interpolate(pred[:, None].float(), size=(half, half), mode='nearest')[0, 0].cpu().numpy().astype(np.uint8)
            if init_candidate:
                confidence[index] = F.interpolate(prob[:, None], size=(half, half), mode='nearest')[0, 0].cpu().numpy()
            else:
                old = np.asarray(Image.open(self.output/'CTR_O'/name))
                new = pred[0].cpu().numpy()
                for c in range(6):
                    union = np.count_nonzero((old == c) | (new == c))
                    stability[index, c] = np.count_nonzero((old == c) & (new == c))/(union+1e-8)
            if (index+1) % 256 == 0:
                logging.info('SND candidates %d/%d', index+1, len(names))
        def threshold(values, ratio):
            values = np.sort(values)
            return float(values[min(int(np.round(len(values)*ratio)), len(values)-1)]) if len(values) else 0.
        thresholds, meta_thresholds = [], []
        for c in range(6):
            values = confidence[labels == c] if init_candidate else stability[:, c][stability[:, c] > 0]
            thresholds.append(threshold(values, cfg.TCR.TOPK_CANDIDATE))
            meta_thresholds.append(threshold(values, .9))
        for bank in ('CTR', 'CTR_O', 'meta_val', 'meta_val_mixup'):
            (self.output/bank).mkdir(parents=True, exist_ok=True)
            if update_meta and bank.startswith('meta_'):
                for path in (self.output/bank).glob('*.png'):
                    path.unlink()
        counts = {}
        histogram = np.zeros(6, dtype=np.int64)
        def save(bank, name, mask):
            Image.fromarray(mask.astype(np.uint8)).resize((self.size, self.size), Image.Resampling.NEAREST).save(self.output/bank/name)
        for index, name in enumerate(names):
            raw = labels[index]
            save('CTR_O', name, raw)
            histogram += np.bincount(raw.ravel(), minlength=6)
            scores = confidence[index] if init_candidate else np.asarray(stability[index])[raw]
            selected = raw.copy()
            selected[scores < np.asarray(thresholds)[raw]] = 255
            scout = getattr(self, 'gsd_supervision', None)
            if hasattr(scout, 'bank_filter'):
                selected = scout.bank_filter(name, selected)
            save('CTR', name, selected)
            counts[name] = int(np.count_nonzero(selected != 255))
            if update_meta:
                meta = selected.copy()  # Original applies meta threshold after CTR filtering.
                meta[scores < np.asarray(meta_thresholds)[raw]] = 255
                if np.mean(meta == 255) <= .98:
                    save('meta_val_mixup', name, meta)
        if update_meta:
            ranked = sorted(names, key=lambda n: counts[n], reverse=True)
            top = max(1, int(len(ranked)*cfg.METAPL.TOP_K))
            for name in ranked[:top]:
                (self.output/'meta_val'/name).write_bytes((self.output/'CTR'/name).read_bytes())
            self.meta_mixer = None
        with (self.output/'candidates.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'iteration': self.last_iteration, 'initial': init_candidate,
                                    'update_meta': update_meta, 'thresholds': thresholds,
                                    'meta_thresholds': meta_thresholds, 'pseudo_fraction': (histogram/histogram.sum()).tolist(),
                                    'meta_files': len(list((self.output/'meta_val').glob('*.png'))),
                                    'meta_mix_files': len(list((self.output/'meta_val_mixup').glob('*.png')))})+'\n')
        encoder.train(modes[0]); classifier.train(modes[1])

    def sample_val_images(self, cfg, pseudo, meta_val='meta_val'):
        with (self.output/(meta_val+'.p')).open('rb') as handle:
            _, file_to_label = pickle.load(handle)
        names = list(file_to_label)
        if not names:
            raise RuntimeError('Empty stable meta bank; do not silently disable METAPL')
        if self.meta_mixer is None:
            self.meta_mixer = ObjectMixer(self, 'meta_val_mixup', meta=True)
        classes = list(range(6))
        present = set(torch.unique(pseudo).cpu().tolist())
        mix_p = np.array([float(c in present) for c in classes])
        if mix_p.sum() == 0:
            mix_p[:] = 1
        images, labels = [], []
        for mask in pseudo:
            current = set(torch.unique(mask).cpu().tolist()) - {255}
            matches = np.array([len(current & set(file_to_label[n])) for n in names], dtype=float)
            p = (matches / matches.max())**3 if matches.max() > 0 else np.ones(len(names))
            name = str(np.random.choice(names, p=p/p.sum()))
            image = self.image(name)
            label = np.asarray(Image.open(self.output/meta_val/name), dtype=np.int64)
            image, pasted = self.meta_mixer.mix(image, classes, mix_p)
            labels.append(torch.from_numpy(np.where(pasted == 255, label, pasted)))
            images.append(normalize(image))
        self.meta_images += len(images)
        return torch.stack(images), torch.stack(labels)

    def record_step(self, iteration, loss, weights, pseudo, pasted, confidence, lr):
        self.last_iteration = iteration
        self.teacher_inferences += pseudo.shape[0]
        if iteration != 1 and iteration % 20:
            return
        if not torch.isfinite(weights).all():
            raise FloatingPointError('Non-finite meta weights')
        record = {'iteration': iteration, 'loss': float(loss.detach()), 'lr_encoder': lr,
                  'lr_classifier': lr*10, 'meta_weight_min': float(weights.min()),
                  'meta_weight_max': float(weights.max()), 'pseudo_valid_fraction': float((pseudo != 255).float().mean()),
                  'paste_fraction': float((pasted != 255).float().mean()), 'running_conf': confidence.detach().cpu().tolist()}
        record['lr_readout'] = lr * getattr(self, 'readout_lr_mult', 10)
        record.update(pseudo_class_pixels=torch.bincount(pseudo[pseudo != 255].flatten(), minlength=6).cpu().tolist(),
                      meta_weight_mean=float(weights.mean()), meta_weight_std=float(weights.std()),
                      meta_weight_q10=float(torch.quantile(weights.float(), .1)),
                      meta_weight_q90=float(torch.quantile(weights.float(), .9)),
                      meta_images=self.meta_images, teacher_inferences=self.teacher_inferences,
                      bank_refreshes=self.bank_refreshes,
                      peak_gpu_memory_mb=(torch.cuda.max_memory_allocated(self.device)/1048576
                                          if self.device.type == 'cuda' else None))
        if getattr(self, 'aux', None) is not None:
            record.update(self.aux.last_record or {})
            if iteration % 500 == 0:
                logging.info('Aux step=%s grad_cosine=%s grad_ratio=%s selected_fraction=%s source_main_agreement=%s',
                             iteration, record.get('aux_grad_cosine'), record.get('aux_grad_ratio'),
                             record.get('aux_selected_fraction'), record.get('aux_source_main_agreement'))
                if 'aux_preparing' in record or 'aux_soft_kl_loss' in record:
                    logging.info('Aux preparation=%s R_update_weight=%s relation_loss=%s soft_kl_loss=%s soft_kl_weight=%s',
                                 record.get('aux_preparing', False), record.get('rebuilder_update_weight'),
                                 record.get('aux_relation_loss'), record.get('aux_soft_kl_loss'),
                                 record.get('aux_soft_kl_weight', 0))
                if 'context_visible_keys' in record:
                    logging.info('Context lambda_nominal=%s lambda_encoder=%s cap_active=%s R_weight=%s decoder_aux_weight=0 keys=%s target_entropy=%s target_information=%s source_energy_fraction=%s teacher_variance=%s',
                        record['aux_weight'], record['aux_encoder_weight'], record['aux_encoder_cap_active'],
                        record['rebuilder_update_weight'], record['context_visible_keys'],
                        record['context_target_entropy'], record['context_target_information'],
                        record['context_source_energy_fraction'], record['context_teacher_variance'])
                if 'contrast_keys' in record:
                    logging.info('Source contrast lambda_nominal=%s lambda_encoder=%s cap_active=%s R_weight=%s decoder_aux_weight=0 grid=%s keys_by_class=%s queries_by_class=%s positive=%s negative=%s margin=%s negative_mass=%s excluded_fraction=%s teacher_confidence=%s',
                        record['aux_weight'], record['aux_encoder_weight'], record['aux_encoder_cap_active'],
                        record['rebuilder_update_weight'], record['contrast_grid'], record['contrast_keys_by_class'],
                        record.get('contrast_queries_by_class'), record.get('contrast_positive_cosine'),
                        record.get('contrast_negative_cosine'), record.get('contrast_positive_margin'),
                        record.get('contrast_negative_mass'), record.get('contrast_excluded_fraction'),
                        record.get('contrast_teacher_confidence'))
                    logging.info('Aux target_teacher=%s requested=%s target_vs_ema_cosine=%s source_agreement=%s probability_l1=%s',
                                 record.get('aux_target_teacher', 'ema'), record.get('aux_target_teacher_requested', 'ema'),
                                 record.get('aux_target_ema_cosine'), record.get('aux_target_ema_source_agreement'),
                                 record.get('aux_target_ema_probability_l1'))
                if record.get('aux_view_disagreement') is not None:
                    logging.info('Aux views disagreement=%s confident_disagreement=%s class_coverage=%s',
                                 record['aux_view_disagreement'], record.get('aux_view_disagreement_confident'),
                                 record.get('aux_selection_coverage_by_teacher_class'))
        if getattr(self, 'gsd_supervision', None) is not None:
            record.update(self.gsd_supervision.last_record)
        if getattr(self, 'dtst_settings', None):
            record.update(self.dtst_last_record)
            record.update(self.dtst_feedback.metrics())
        record.update(getattr(self, 'geometry_last_record', {}))
        record.update(getattr(self, 'context_last_record', {}))
        if getattr(self, 'semantic_recovery', None) is not None:
            record.update(self.semantic_recovery.last_record)
        if getattr(self, 'baseline', None) is not None:
            record.update(self.baseline.last_record)
        with (self.output/'training_history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record)+'\n')

    def checkpoint(self, iteration, encoder, classifier):
        if getattr(self, 'freeze_source_readout', False):
            for name, expected in self.source_readout.items():
                if not torch.equal(classifier.state_dict()[name].detach().cpu(), expected):
                    raise RuntimeError('Source readout changed despite frozen-source policy')
        state = {'iteration': iteration, 'feature_extractor': encoder.state_dict(),
                 'classifier': classifier.state_dict(), 'protocol_signature': self.signature,
                 'task': 'vh2pd', 'method': getattr(self, 'method', 'SND'),
                 'stage': self.output.name, 'num_classes': 6}
        if self.architecture == 'segformer_mit_b2_nvidia_mlp768':
            state.update(architecture=self.architecture, protocol='siamseg_sfuda_b2_v1',
                         source_checkpoint_sha256=getattr(self, 'source_sha', None),
                         smoke_only=getattr(self, 'smoke_only', False),
                         total_iteration=iteration + getattr(self, 'stage_offset', 0))
        path = self.output/'last.pth'
        temporary = path.with_suffix('.pth.tmp')
        torch.save(state, temporary)
        temporary.replace(path)
        self.record_evaluation(iteration, encoder, classifier)
        if getattr(self, 'aux', None) is not None and self.output.name == 'warm':
            peak = json.loads((self.output/'observed_peak.json').read_text())
            if peak['iteration'] == iteration:
                anchor = {'iteration': iteration, 'protocol_signature': self.signature,
                          'source_checkpoint_sha256': self.source_sha, 'aux': self.aux.state_dict()}
                temporary = self.output/'best_aux_state.pth.tmp'
                torch.save(anchor, temporary)
                temporary.replace(self.output/'best_aux_state.pth')
        if getattr(self, 'checkpoint_portable', None):
            self.checkpoint_portable()

    def save_training_state(self, iteration, student, decoder, teacher, teacher_decoder,
                            optimizer, decoder_optimizer, running_conf, initial_bank):
        state = {'iteration': iteration, 'stage': self.output.name,
                 'protocol_signature': self.signature, 'source_checkpoint_sha256': self.source_sha,
                 'smoke_only': self.smoke_only,
                 'student_feature_extractor': student.state_dict(), 'student_classifier': decoder.state_dict(),
                 'feature_extractor': teacher.state_dict(), 'classifier': teacher_decoder.state_dict(),
                 'optimizer_feature_extractor': optimizer.state_dict(),
                 'optimizer_classifier': decoder_optimizer.state_dict(),
                 'running_conf': running_conf.detach().cpu(), 'initial_bank': initial_bank,
                 'rng': rng_state(), 'teacher_inferences': self.teacher_inferences,
                 'bank_refreshes': self.bank_refreshes, 'meta_images': self.meta_images}
        if getattr(self, 'aux', None) is not None:
            state['aux'] = self.aux.state_dict()
        if getattr(self, 'gsd_supervision', None) is not None:
            state['gsd_extra_teacher_images'] = self.gsd_supervision.extra_forwards
        if getattr(self, 'dtst_settings', None):
            state['dtst_feedback'] = self.dtst_feedback.state_dict()
        if getattr(self, 'semantic_recovery', None) is not None:
            state['semantic_recovery'] = self.semantic_recovery.state_dict()
        if getattr(self, 'baseline', None) is not None:
            state['baseline'] = self.baseline.state_dict()
        path = self.output/'resume.pth'
        temporary = path.with_suffix('.pth.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    def record_evaluation(self, iteration, encoder, classifier):
        if (not self.eval_period or iteration in self.evaluated_iterations or
                (iteration % self.eval_period and iteration != self.stop_iteration)):
            return
        from scripts.diagnose_crots_remote import EvaluationImages, evaluate_at_size
        random_state = rng_state()
        modes = encoder.training, classifier.training
        try:
            encoder.eval(); classifier.eval()
            dataset = EvaluationImages(self.root, self.eval_split, 6, 0, self.eval_limit)
            forward = lambda images, size: classifier(encoder(images)[1], size)
            metrics = evaluate_at_size(forward, dataset, 6, self.device, self.workers, False)
        finally:
            encoder.train(modes[0]); classifier.train(modes[1])
            restore_rng(random_state)
        point = {'iteration': iteration, 'total_iteration': iteration + getattr(self, 'stage_offset', 0),
                 'stage': self.output.name, 'metrics': metrics,
                 'diagnostic_only': self.architecture != 'segformer_mit_b2_nvidia_mlp768',
                 'validation_selected': self.architecture == 'segformer_mit_b2_nvidia_mlp768',
                 'protocol_signature': self.signature,
                 'image_count': len(dataset), 'model': 'EMA', 'used_for_training_or_stopping': False}
        if getattr(self, 'freeze_source_readout', False):
            point['readout_source_max_abs_delta'] = max(float((classifier.state_dict()[name].detach().cpu() - value).abs().max())
                                                        for name, value in self.source_readout.items())
            logging.info('Frozen source readout at step%d: max_abs_delta=%g', iteration,
                         point['readout_source_max_abs_delta'])
        path = self.output/'evaluation_curve.jsonl'
        miou5_path = self.output/'observed_peak_miou5.json'
        if miou5_path.exists():
            previous_peak5 = json.loads(miou5_path.read_text())
        else:
            earlier = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            previous_peak5 = max(earlier, key=lambda entry: entry['metrics']['miou5']) if earlier else None
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(point)+'\n')
        peak_path = self.output/'observed_peak.json'
        previous_peak = json.loads(peak_path.read_text()) if peak_path.exists() else None
        new_peak6 = previous_peak is None or metrics['miou'] > previous_peak['metrics']['miou']
        if new_peak6:
            write_json(peak_path, dict(point, selection=('target_validation_selected' if
                      self.architecture == 'segformer_mit_b2_nvidia_mlp768' else
                      'Oracle maximum over target-eval measurements; diagnostic only')))
            if self.architecture == 'segformer_mit_b2_nvidia_mlp768' and (self.output/'last.pth').is_file():
                shutil.copy2(self.output/'last.pth', self.output/'best_target_val.pth')
        new_peak5 = previous_peak5 is None or metrics['miou5'] > previous_peak5['metrics']['miou5']
        if new_peak5:
            write_json(miou5_path, dict(point, peak_metric='miou5',
                                        selection='diagnostic peak; checkpoint selection uses mIoU6'))
        elif not miou5_path.exists():
            write_json(miou5_path, dict(previous_peak5, peak_metric='miou5',
                                        selection='diagnostic peak; checkpoint selection uses mIoU6'))
        self.evaluated_iterations.add(iteration)
        def percent(value):
            return 'nan' if value is None else f'{100*value:.2f}'
        rows = '\n'.join(
            f'  {name}: IoU {percent(metrics["iou_by_class"][name])}  '
            f'Acc {percent(metrics["accuracy_by_class"][name])}  '
            f'F1 {percent(metrics["f1_by_class"][name])}  '
            f'Precision {percent(metrics["precision_by_class"][name])}  '
            f'Recall {percent(metrics["recall_by_class"][name])}'
            for name in metrics['class_order'])
        prior6 = ('none' if previous_peak is None else
                  f'{100*previous_peak["metrics"]["miou"]:.2f}% at step {previous_peak["iteration"]}')
        prior5 = ('none' if previous_peak5 is None else
                  f'{100*previous_peak5["metrics"]["miou5"]:.2f}% at step {previous_peak5["iteration"]}')
        best6 = point if new_peak6 else previous_peak
        best5 = point if new_peak5 else previous_peak5
        logging.info('%s curve step %d: mIoU6 %.2f%%, mIoU5 %.2f%%, peak mIoU6 %.2f%% at step %d, peak mIoU5 %.2f%% at step %d, previous peak mIoU6 %s, previous peak mIoU5 %s, aAcc %.2f%%, mAcc %.2f%%, mF1 %.2f%%, mPrecision %.2f%%, mRecall %.2f%%\n%s',
                     self.output.name, iteration, 100*metrics['miou'], 100*metrics['miou5'],
                     100*best6['metrics']['miou'], best6['iteration'],
                     100*best5['metrics']['miou5'], best5['iteration'], prior6, prior5,
                     100*metrics['overall_accuracy'], 100*metrics['mean_accuracy'],
                     100*metrics['mean_f1'], 100*metrics['mean_precision'], 100*metrics['mean_recall'], rows)

