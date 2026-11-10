"""Prepare/audit native SiamSeg configs and export metrics, never our SRMP model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import pprint
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '6554c9e92b4de157eae9d1d59d00ee077b30ace3'
CONFIGS = {
    'r2u': 'siamseg_daformer_sepaspp_mitb5_512x512_40k_Rural_2_Urban.py',
    'vh2pd': 'siamseg_daformer_sepaspp_mitb5_512x512_40k_VaihingenIRRG_2_PotsdamRGB.py',
}
CLASSES = {
    'r2u': ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural'),
    'vh2pd': ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter'),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_upstream(repo):
    manifest = json.loads((ROOT / 'configs/siamseg_upstream_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        actual = hashlib.sha256((repo / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise ValueError(f'Public code differs from pinned revision: {name}')
    if (repo / '.git').exists():
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != COMMIT:
            raise ValueError(f'Unexpected upstream revision: {revision}')
    return manifest


def build_config(original, *, task, stage, data_root, backbone, work_dir,
                 samples_per_gpu=6, workers=3):
    cfg = copy.deepcopy(original)
    cfg['work_dir'] = str(work_dir)
    cfg['model']['train_cfg']['work_dir'] = str(work_dir)
    cfg['model']['pretrained'] = str(backbone)
    cfg['data']['samples_per_gpu'] = samples_per_gpu
    cfg['data']['workers_per_gpu'] = workers
    root = str(data_root / 'LoveDA') if task == 'r2u' else str(data_root)
    for role in ('source', 'target'):
        cfg['data']['train'][role]['data_root'] = root
    for role in ('val', 'test'):
        cfg['data'][role]['data_root'] = root
    cfg['load_from'] = None
    cfg['resume_from'] = None
    # The authors' UDA is joint training from ImageNet, not source -> SFDA.
    if stage in ('source', 'source-segformer'):
        cfg.pop('uda')
        cfg['data']['train'] = cfg['data']['train']['source']
        cfg['use_ddp_wrapper'] = False
        # UDA backpropagates itself. Source-only needs the native optimizer hook.
        cfg['optimizer_config'] = {}
        if stage == 'source-segformer':
            head = cfg['model']['decode_head']
            cfg['model']['decode_head'] = {
                'type': 'SegformerHead', 'in_channels': head['in_channels'],
                'in_index': head['in_index'], 'channels': 256,
                'dropout_ratio': head['dropout_ratio'], 'num_classes': head['num_classes'],
                'norm_cfg': head['norm_cfg'], 'align_corners': False,
                'loss_decode': head['loss_decode'],
            }
    elif stage == 'dacs':
        cfg['uda']['type'] = 'DACS'
        cfg['uda'].pop('contras_model_cfg', None)
    return cfg


def prepare(args):
    import torch
    import mmcv
    import numpy as np
    import PIL
    from mmcv import Config
    verify_upstream(args.repo)
    if args.samples_per_gpu < 2 and args.stage == 'adapt':
        raise ValueError('SimSiam BatchNorm1d requires at least two samples on each GPU')
    original_path = args.repo / 'configs/SiamSeg' / CONFIGS[args.task]
    original = Config.fromfile(str(original_path))._cfg_dict.to_dict()
    cfg = build_config(original, task=args.task, stage=args.stage, data_root=args.data_root,
                       backbone=args.backbone, work_dir=args.work_dir,
                       samples_per_gpu=args.samples_per_gpu, workers=args.workers)
    preprocess = json.loads((args.data_root / f'preprocess_{args.task}.json').read_text(encoding='utf-8'))
    notes = [
        'Public config is authoritative: EMA=0.999, pseudo_threshold=0.968; paper differs.',
        'Official UDA starts jointly from ImageNet; it does not load our source checkpoint.',
        'Public target preprocessing reads GT for category-constrained crop; this is not label-free preprocessing.',
        'Selection fixed to iter_40000; validation disabled during train; no target-based checkpoint selection.',
        'Auxiliary package versions are pinned locally; upstream did not specify all versions.',
    ]
    if args.stage in ('source', 'source-segformer'):
        notes.append('Source-only config reconstructed: no task-specific source training config in authors public repo.')
    if args.stage == 'source':
        notes.append('Same DAFormer decoder control; not the published SegFormer source row.')
    if args.stage == 'source-segformer':
        notes.append('SegFormer source reconstruction; decoder width/loss/schedule not confirmed by authors.')
    if args.task == 'vh2pd':
        notes.append('ISPRS counts differ from paper; native public tile groups with repaired stride; paper reproduction unverified.')
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    path = work / 'generated_config.py'
    # Python config preserves tuple types required by MMSeg's Resize/TTA.
    encoded = '\n\n'.join(f'{key} = {pprint.pformat(value, width=100, sort_dicts=False)}'
                           for key, value in cfg.items()) + '\n'
    protocol_path = work / 'protocol_reproduce.json'
    protocol = {
        'task': args.task, 'stage': args.stage, 'upstream_commit': COMMIT,
        'upstream_config': CONFIGS[args.task], 'source_config_sha256': digest(original_path),
        'effective_config': cfg, 'preprocess': preprocess, 'seed': args.seed,
        'world_size': args.gpus, 'source_images_per_update': args.gpus * args.samples_per_gpu,
        'target_images_per_update': args.gpus * args.samples_per_gpu if args.stage in ('adapt', 'dacs') else 0,
        'target_labels_used_in_loss': False,
        'target_labels_used_in_preprocessing': args.stage in ('adapt', 'dacs'),
        'backbone_sha256': digest(args.backbone),
        'versions': {'python': sys.version, 'torch': str(torch.__version__),
                     'torch_cuda': torch.version.cuda, 'mmcv': mmcv.__version__,
                     'numpy': np.__version__, 'pillow': PIL.__version__},
        'selection_rule': 'last scheduled iter_40000; no target validation for selection',
        'paper_protocol_verified': False, 'notes': notes,
    }
    protocol = json.loads(json.dumps(protocol))
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding='utf-8'))
        if old != protocol:
            raise ValueError('Run directory already has a different protocol; choose a separate RESULT_ROOT')
    path.write_text(encoded, encoding='utf-8')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    print(path)


def check(args):
    import torch
    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    verify_upstream(args.repo)
    cfg = Config.fromfile(str(args.work_dir / 'generated_config.py'))
    train = build_dataset(cfg.data.train)
    test = build_dataset(dict(cfg.data.test, test_mode=True))
    if tuple(test.CLASSES) != CLASSES[args.task]:
        raise ValueError('Dataset class order mismatch')
    # Validate every required native mask/image path without reading target GT in our method.
    components = (train.source, train.target, test) if hasattr(train, 'source') else (train, test)
    if args.task == 'r2u':
        counts = [len(dataset) for dataset in components]
        expected_counts = [1366, 1156, 677] if hasattr(train, 'source') else [1366, 677]
        if counts != expected_counts:
            raise ValueError(f'LoveDA split counts {counts} != {expected_counts}')
    for dataset in components:
        if not dataset.img_infos:
            raise ValueError('Empty dataset')
        for row in dataset.img_infos:
            for path in (Path(dataset.img_dir) / row['filename'],
                         Path(dataset.ann_dir) / row['ann']['seg_map']):
                if not path.is_file():
                    raise FileNotFoundError(path)
        sample = dataset[0]
        print(f'{dataset.__class__.__name__}: {len(dataset)} samples; first keys {list(sample)}')
    # Exact backbone key/shape coverage, before silently accepting a partial pretrained load.
    architecture = copy.deepcopy(cfg.model)
    architecture.pretrained = None
    model = build_segmentor(architecture)
    payload = torch.load(str(args.backbone), map_location='cpu')
    weights = payload.get('state_dict', payload.get('model', payload))
    # Match the upstream loader's default key revision, not a broader remap
    # that would pass this check but fail to load in the original trainer.
    weights = {key[len('module.'):] if key.startswith('module.') else key: value
               for key, value in weights.items()}
    expected = model.backbone.state_dict()
    missing = [key for key, tensor in expected.items()
               if key not in weights or tuple(weights[key].shape) != tuple(tensor.shape)]
    if missing:
        raise ValueError(f'Pretrained MiT-B5 has missing/incompatible backbone keys: {missing[:12]}')
    print(f'Backbone verified: {len(expected)} keys/shapes; no training performed')
    print(f'CUDA visible: {torch.cuda.device_count()} GPUs')


def report(args):
    import numpy as np
    work = args.work_dir
    protocol = json.loads((work / 'protocol_reproduce.json').read_text(encoding='utf-8'))
    # This file is written by our call to the upstream evaluator, never an input download.
    with (work / 'eval_pre_eval.pkl').open('rb') as stream:
        rows = pickle.load(stream)
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError('Expected upstream pre_eval tuples: intersection, union, prediction, ground truth')
    # Match MMSeg's float64 aggregation across all validation images.
    areas = [np.stack([np.asarray(row[i], dtype=np.float64) for row in rows]).sum(axis=0)
             for i in range(4)]
    intersection, union, predicted, gt = areas
    if len(union) != len(CLASSES[args.task]):
        raise ValueError('Class count mismatch in evaluation')
    iou = np.divide(intersection, union, out=np.full(len(union), np.nan), where=union > 0)
    f1 = np.divide(2 * intersection, predicted + gt, out=np.full(len(union), np.nan), where=predicted + gt > 0)
    metrics = {
        'miou': float(np.nanmean(iou)), 'mf1': float(np.nanmean(f1)),
        'iou_by_class': {key: float(v) if np.isfinite(v) else None for key, v in zip(CLASSES[args.task], iou)},
        'f1_by_class': {key: float(v) if np.isfinite(v) else None for key, v in zip(CLASSES[args.task], f1)},
        'valid_pixels': int(gt.sum()), 'evaluation_images': len(rows),
        'intersection': intersection.tolist(), 'union': union.tolist(),
        'prediction_area': predicted.tolist(), 'ground_truth_area': gt.tolist(),
    }
    if args.task == 'vh2pd':
        metrics['miou5'] = float(np.nanmean(iou[:5]))
    reference = {'miou': 0.5172 if args.stage == 'adapt' else 0.483} if args.task == 'r2u' and args.stage in ('adapt', 'dacs') else None
    if args.task == 'vh2pd' and args.stage in ('adapt', 'dacs', 'source-segformer'):
        reference = {'miou': {'adapt': 0.5574, 'dacs': 0.5446, 'source-segformer': 0.4108}[args.stage]}
    result = {
        'schema_version': 1, 'contains_weights': False, 'task': args.task,
        'method': args.stage, 'metrics_units': 'fractions', 'metrics': metrics,
        'protocol': protocol, 'evaluated_model': 'student (upstream evaluator)',
        'checkpoint_sha256': digest(args.checkpoint), 'config_sha256': digest(work / 'generated_config.py'),
        'published_reference': reference,
        'published_reference_status': 'context only; exact source/config/batch/preprocess equivalence unverified',
    }
    if reference:
        result['delta_published_miou_percentage_points'] = 100 * (metrics['miou'] - reference['miou'])
    if args.stage in ('adapt', 'dacs'):
        before = work.parent / 'source/result_portable.json'
        if before.is_file():
            source = json.loads(before.read_text(encoding='utf-8'))
            if source['protocol']['backbone_sha256'] == protocol['backbone_sha256'] and source['protocol']['preprocess'] == protocol['preprocess']:
                result['source_control_comparison'] = {
                    'control': 'same DAFormer architecture; independent joint-UDA initialization',
                    'miou_before': source['metrics']['miou'],
                    'delta_miou_percentage_points': 100 * (metrics['miou'] - source['metrics']['miou']),
                }
    (work / 'result_portable.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print(f'Portable result: {work / "result_portable.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'report'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--task', choices=tuple(CONFIGS), required=True)
    parser.add_argument('--stage', choices=('source', 'source-segformer', 'dacs', 'adapt'), required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--backbone', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--samples-per-gpu', type=int, default=6)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.gpus < 1 or args.samples_per_gpu < 1:
        parser.error('GPU count and batch must be positive')
    if args.action in ('prepare', 'check') and (not args.data_root or not args.backbone):
        parser.error('--data-root and --backbone required')
    sys.path.insert(0, str(args.repo))
    {'prepare': prepare, 'check': check, 'report': report}[args.action](args)


if __name__ == '__main__':
    main()
