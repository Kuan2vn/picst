"""Evaluate saved VH->PD source checkpoints; never train or replace weights."""
import argparse
import json
from pathlib import Path
import zipfile

import torch

from train_SND_remote import evaluate_checkpoint, seed_all
from train_SRMP_remote import load_checkpoint, sha256_file
from scripts.snd_remote import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True, type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('data/remote_sensing/dgle'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    args.eval_limit = 0
    torch.set_num_threads(4)
    seed_all(1)
    keys = ('task', 'stage', 'num_classes', 'source_split', 'source_manifest_sha256',
            'backbone_sha256', 'train_policy', 'normalization', 'seed', 'crop_size')
    common, checkpoints = None, []
    for path in args.checkpoints:
        state = load_checkpoint(path)
        metadata = {k: state[k] for k in keys}
        if metadata['task'] != 'vh2pd' or metadata['stage'] != 'source':
            raise ValueError('Expected VH->PD source checkpoints')
        if common is not None and common != metadata:
            raise ValueError('Checkpoint source protocols differ; do not merge their curves')
        common = metadata
        checkpoints.append((int(state['iteration']), path, sha256_file(path)))
        del state
    if len({step for step, _, _ in checkpoints}) != len(checkpoints):
        raise ValueError('Duplicate source iteration')
    if common['source_manifest_sha256'] != sha256_file(args.data_root/'manifests/isprs_vh_irrg_train.csv'):
        raise ValueError('Source manifest mismatch')
    report = {'task': 'vh2pd', 'diagnostic_only': True, 'weights_changed': False,
              'scope': 'Target-eval oracle peak among saved source checkpoints, not a deployable selection rule',
              'split': 'isprs_pd_rgb_eval', 'source_metadata': common,
              'manifest_sha256': sha256_file(args.data_root/'manifests/isprs_pd_rgb_eval.csv'),
              'torch': str(torch.__version__), 'amp': False, 'points': []}
    for step, path, digest in sorted(checkpoints):
        metrics = evaluate_checkpoint(path, args, 'isprs_pd_rgb_eval')
        if report['points'] and metrics['valid_pixels'] != report['points'][0]['metrics']['valid_pixels']:
            raise ValueError('Evaluation pixel count changed')
        report['points'].append({'iteration': step, 'checkpoint': str(path), 'sha256': digest, 'metrics': metrics})
        peak = max(report['points'], key=lambda p: p['metrics']['miou'])
        report['observed_oracle_peak'] = {'iteration': peak['iteration'], 'miou': peak['metrics']['miou']}
        write_json(args.output/'source_curve.json', report)
        with zipfile.ZipFile(args.output/'source_curve_portable.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(args.output/'source_curve.json', 'source_curve.json')
        print(f"source step={step} mIoU={metrics['miou']*100:.4f}%", flush=True)


if __name__ == '__main__':
    main()
