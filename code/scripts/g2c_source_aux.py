"""G2C adapters for the shared source ReMask/MRM-relational trainer.

Uses the latest source-init G2C SLURM's R101 config, source20k and 30k budget.
Training reads RGB only: phase1 Entropy/KL ReMask by default, without TCR/meta/mixing.
"""
from pathlib import Path
import json
import logging
import random
import shutil

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from core.configs import cfg as base_config
from core.datasets.build import build_transform
from core.models.mrm_rebuilder import Rebuilder
from train_SRMP import source_kernel_geometry
from train_SRMP_remote import make_model, load_checkpoint, sha256_file, rng_state, restore_rng
from scripts.snd_srmp import RelationalAux
from scripts.snd_remote import write_json
from scripts.diagnose_crots_remote import summarize_matrix

SOURCE_SHA = 'be51c0a1132255e124f7539cd85139d3c762e28723a666b108a35cef6ccf7cde'
PROTOCOL_SLURM = Path(__file__).resolve().parents[1] / 'gta5_to_cityscapes_srmp.slurm'
CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
           'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'train', 'motorcycle', 'bicycle')
RAW_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
LOG = logging.getLogger('dgle')


def config(args):
    cfg = base_config.clone()
    cfg.merge_from_file(str(args.city_config))
    return cfg


def paths(args, split):
    listing = args.city_train_list if split == 'train' else args.city_val_list
    names = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f'Duplicate Cityscapes {split} images')
    root = args.city_root
    image_dir = 'leftImg8bit' if (root / 'leftImg8bit').is_dir() else 'data'
    label_dir = 'gtFine' if (root / 'gtFine').is_dir() else 'label'
    rows = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not name.endswith('_leftImg8bit.png'):
            raise ValueError(f'Invalid Cityscapes list entry: {name}')
        image = root / image_dir / split / relative
        label = root / label_dir / split / name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        if not image.is_file() or (split == 'val' and not label.is_file()):
            raise FileNotFoundError(image if not image.is_file() else label)
        rows.append((image, label))
    expected = 2975 if split == 'train' else 500
    if not args.smoke and len(rows) != expected:
        raise ValueError(f'Expected all {expected} Cityscapes {split} images; found {len(rows)}')
    limit = args.data_limit if split == 'train' else args.eval_limit
    return rows[:limit] if limit else rows


def preflight_city(args, source):
    if any(path is None for path in (args.city_root, args.city_train_list, args.city_val_list)):
        raise ValueError('Provide city root and the previously used train/validation lists')
    if sha256_file(args.source) != SOURCE_SHA:
        raise ValueError('Expected G2C source20k from gta5_to_cityscapes_srmp.slurm')
    for i in range(4):
        if source['classifier'][f'conv2d_list.{i}.weight'].shape != (19, 2048, 3, 3):
            raise ValueError('Source ASPP architecture mismatch')
    cfg = config(args)
    if (cfg.MODEL.NAME != 'deeplab_resnet101' or not cfg.MODEL.FREEZE_BN or cfg.MODEL.NUM_CLASSES != 19):
        raise ValueError('Expected the prior frozen-BN DeepLabV2 ResNet101 G2C configuration')
    train, val = paths(args, 'train'), paths(args, 'val')
    return dict(reference_slurm=PROTOCOL_SLURM.name, reference_slurm_sha256=sha256_file(PROTOCOL_SLURM),
        source_training_step=20000, reference_adaptation_updates=30000,
        city_config_sha256=sha256_file(args.city_config),
        train_list_sha256=sha256_file(args.city_train_list), val_list_sha256=sha256_file(args.city_val_list),
        train_images=len(train), validation_images=len(val),
        canonical_size=[1536, 768], crop_size=list(cfg.INPUT.TARGET_INPUT_SIZE_TRAIN),
        scale_range=list(cfg.INPUT.INPUT_SCALES_TRAIN), validation_input_size=list(cfg.INPUT.INPUT_SIZE_TEST),
        ignore_label=255, class_order=list(CLASSES), evaluation_model='student',
        differences_from_reference_slurm=f'current MRM aux and trainable main head; ReMask {args.remask_phase} reliability without TCR/meta/mixing/prototypes; uniform sampling; student validation; BF16 opt-in')


class Segmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.classifier = make_model(19, freeze_bn=True)

    def forward(self, image, size=None):
        return self.classifier(self.encoder(image)[1], size or image.shape[-2:])


def load_model(path, device):
    saved = load_checkpoint(path)
    model = Segmentation()
    model.encoder.load_state_dict(saved['feature_extractor'], strict=True)
    model.classifier.load_state_dict(saved['classifier'], strict=True)
    return model.to(device)


def align_city(tensor, geometry, mode='bilinear'):
    outputs = []
    for i in range(tensor.shape[0]):
        h, w = (int(v[i]) for v in geometry['rescale_size'])
        y1, y2, x1, x2 = (int(v[i]) for v in geometry['random_crop_axis'])
        kwargs = dict(align_corners=True) if mode == 'bilinear' else {}
        item = F.interpolate(tensor[i:i+1].float(), (h, w), mode=mode, **kwargs)
        item = item[:, :, y1:y2, x1:x2]
        if bool(geometry['RandomHorizontalFlip'][i]):
            item = item.flip(-1)
        outputs.append(item)
    return torch.cat(outputs)


class TrainingImages(Dataset):
    def __init__(self, args):
        self.rows = paths(args, 'train')
        cfg = config(args)
        # Full canonical image and crop match cityscapes_meta's previous protocol.
        self.transform = build_transform(cfg, 'train', False)
        self.mean = torch.tensor(cfg.INPUT.PIXEL_MEAN)[:, None, None]
        self.std = torch.tensor(cfg.INPUT.PIXEL_STD)[:, None, None]
        minimum_scale = min(cfg.INPUT.INPUT_SCALES_TRAIN)
        if (cfg.INPUT.TO_BGR255 or minimum_scale * 768 < cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1] or
                minimum_scale * 1536 < cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0]):
            raise ValueError('G2C adapter expects RGB normalization and geometry without padding')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, token):
        index, augmentation_seed = token
        with Image.open(self.rows[index][0]) as loaded:
            image = loaded.convert('RGB').resize((1536, 768), Image.Resampling.BILINEAR)
        # Dummy mask conveys geometry only; target training annotations are never opened.
        dummy = Image.new('L', image.size, 255)
        saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(augmentation_seed); np.random.seed(augmentation_seed % 2**32)
            torch.random.default_generator.manual_seed(augmentation_seed)
            strong, _, geometry = self.transform(image, dummy)
        finally:
            random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
        full = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        full = (full - self.mean) / self.std
        batched_geometry = {key: [torch.tensor([x]) for x in value] if isinstance(value, (list, tuple))
                            else torch.tensor([value]) for key, value in geometry.items()}
        clean = align_city(full[None], batched_geometry)[0]
        return strong, torch.full(strong.shape[-2:], 255, dtype=torch.long), clean, full, geometry, index


class ASPPAux(RelationalAux):
    """Use the same schedule/RNG/checkpoint interface, with official DLV2 MRM."""
    def __init__(self, classifier, settings, device, seed, base_lr):
        self.device, self.settings = torch.device(device), dict(settings)
        self.directions, gram = source_kernel_geometry(classifier)
        self.gram = gram.to(device)
        # Anchors are recorded exactly so resume cannot silently change geometry.
        self.weight = self.directions.to(device)
        self.bias = sum(conv.bias.detach() for conv in classifier.conv2d_list)
        self.mask_rng = torch.Generator(device=self.device).manual_seed(seed + 80017)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 70017)
            self.rebuilder = Rebuilder(dict(type='DLV2Head', in_channels=2048, in_index=[0],
                embed_dim=settings['hidden_dim'], depth=settings['depth'], mask_ratio=settings['mask_ratio'])).to(device)
            self.aux_cpu_rng = torch.get_rng_state()
            self.aux_cuda_rng = torch.cuda.get_rng_state(self.device) if devices else None
        self.optimizer = torch.optim.AdamW(self.rebuilder.parameters(), lr=base_lr * 10,
                                          weight_decay=settings['weight_decay'])
        self.forward_count = 0


class ValidationImages(Dataset):
    def __init__(self, args):
        self.rows = paths(args, 'val')
        self.transform = build_transform(config(args), 'val', False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path, label_path = self.rows[index]
        with Image.open(image_path) as loaded:
            image = loaded.convert('RGB')
        with Image.open(label_path) as loaded:
            raw = np.asarray(loaded).copy()
        labels = np.full(raw.shape, 255, dtype=np.uint8)
        for train_id, raw_id in enumerate(RAW_IDS):
            labels[raw == raw_id] = train_id
        image, _, _ = self.transform(image, Image.fromarray(labels))
        return image, torch.from_numpy(labels.astype(np.int64))


@torch.inference_mode()
def evaluate_city(args, model, round_number, iteration, output, signature):
    curve = output / 'evaluation_curve.jsonl'
    if curve.exists() and any(json.loads(line)['iteration'] == iteration for line in curve.read_text().splitlines()):
        return
    modes, saved_rng = model.training, rng_state()
    matrix = torch.zeros(19, 19, dtype=torch.int64)
    dataset = ValidationImages(args)
    try:
        model.eval()
        for image, labels in DataLoader(dataset, batch_size=1, num_workers=args.workers):
            logits = model(image.to(args.device), size=labels.shape[-2:])
            prediction = logits.argmax(1).cpu()
            valid = labels != 255
            matrix += torch.bincount(labels[valid] * 19 + prediction[valid], minlength=361).reshape(19, 19)
    finally:
        model.train(modes); restore_rng(saved_rng)
    metrics = summarize_matrix(matrix.numpy(), CLASSES)
    point = dict(iteration=iteration, metrics=metrics, image_count=len(dataset), model='student',
        protocol_signature=signature, target_validation_used_for_reporting=True,
        used_for_training_or_stopping=False, smoke_only=args.smoke)
    with curve.open('a') as stream:
        stream.write(json.dumps(point) + '\n')
    peak_path = output / 'observed_peak.json'
    previous = json.loads(peak_path.read_text()) if peak_path.exists() else None
    if previous is None or metrics['miou'] > previous['metrics']['miou']:
        write_json(peak_path, point)
        shutil.copy2(output / 'last.pth', output / 'best_target_val.pth')
    peak = json.loads(peak_path.read_text())
    LOG.info('G2C step%d: mIoU19 %.2f%%; peak %.2f%% at step%d; previous peak %s; aAcc %.2f%% mAcc %.2f%% mF1 %.2f%% mPrecision %.2f%% mRecall %.2f%%',
        iteration, 100 * metrics['miou'], 100 * peak['metrics']['miou'], peak['iteration'],
        'none' if previous is None else f'{100 * previous["metrics"]["miou"]:.2f}% at step{previous["iteration"]}',
        100 * metrics['overall_accuracy'], 100 * metrics['mean_accuracy'], 100 * metrics['mean_f1'],
        100 * metrics['mean_precision'], 100 * metrics['mean_recall'])
    for name in CLASSES:
        LOG.info('  %s: IoU %s Acc %s F1 %s Precision %s Recall %s', name,
            *(None if metrics[key][name] is None else round(100 * metrics[key][name], 2)
              for key in ('iou_by_class', 'accuracy_by_class', 'f1_by_class', 'precision_by_class', 'recall_by_class')))
