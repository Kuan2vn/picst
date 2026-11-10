#!/usr/bin/env python3
"""One entry point for every method in this repository (VH IRRG -> PD, SegFormer-B2, source-free).

Examples
  python run.py --method picst --setting irrg            # our method (revised PIC-ST) on VH IRRG -> PD IRRG file
  python run.py --method picst --setting rgb             # our method on VH IRRG -> PD RGB
  python run.py --method dtst  --setting irrg            # host baseline (DT-ST)
  python run.py --method s3t_sfda --setting irrg         # re-implemented S3T-SFDA (TGRS)
  python run.py --method stcgn --setting irrg            # re-implemented St-CGN (TGRS 2026), incl. source warm-up
  python run.py --method eval --checkpoint runs/.../warm/best_target_val.pth --setting irrg
Every training run uses the same protocol: source40k, batch 6, seed 1, AdamW 2e-5 (decoder x10, readout x1),
LR horizon 32000, evaluation every 1000 updates, early stop after 2 evaluations without a new mIoU6 peak,
cap 9000 updates (continues while the cap evaluation is a new peak, hard max 20000).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE/'code'
SETTINGS = CODE/'kaggle_settings'

# method -> (runner, experiment per setting); runner 'pheno' = PIC-ST family, 'baseline' = other methods
METHODS = {
    'dtst':      ('pheno', {'irrg': 'pheno_null', 'rgb': 'pheno_null'}),            # DT-ST (CVPR'23) host
    'picst':     ('pheno', {'irrg': 'pheno_c0_c1plaus', 'rgb': 'pheno_c0multi_c1plaus'}),  # ours (revised)
    'picst_v1':  ('pheno', {'irrg': 'pheno_c0_c1', 'rgb': 'pheno_c0_c1'}),           # ours, first version
    'picst_c0':  ('pheno', {'irrg': 'pheno_c0', 'rgb': 'pheno_c0multi'}),            # ablation: C0 only
    'picst_c1':  ('pheno', {'irrg': 'pheno_c1', 'rgb': 'pheno_c1'}),                 # ablation: C1 only
    'snd':       ('baseline', {'irrg': 'snd_author'}),                                # SND (CVPR'24), author pipeline
    'stable_st': ('baseline', {'irrg': 'stable_st'}),                                 # Stable-ST (IJCV'26)
    's3t_sfda':  ('baseline', {'irrg': 's3t_sfda'}),                                  # S3T-SFDA (TGRS), re-implemented
    'stcgn':     ('baseline', {'irrg': 'stcgn'}),                                     # St-CGN (TGRS'26), re-implemented
}
ROOTS = {'irrg': 'gsd_sfuda_b2_irrg_v1', 'rgb': 'siamseg_sfuda_b2_v1'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True, choices=sorted(METHODS) + ['eval'])
    ap.add_argument('--setting', choices=('irrg', 'rgb'), default='irrg',
                    help='irrg: VH IRRG -> PD IRRG file (main); rgb: VH IRRG -> PD RGB')
    ap.add_argument('--data-dir', type=Path, default=HERE/'data', help='folder filled by download_data.py')
    ap.add_argument('--output-dir', type=Path, default=HERE/'runs')
    ap.add_argument('--checkpoint', type=Path, help='--method eval only')
    ap.add_argument('--hours', type=float, default=1e6, help='optional wall-clock budget; pauses at the next evaluation')
    a = ap.parse_args()
    data, source = a.data_dir.resolve(), (a.data_dir/'source_40000.pth').resolve()
    root = data/ROOTS[a.setting]
    env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    if a.method == 'eval':
        cmd = [sys.executable, '-u', str(HERE/'eval_checkpoint.py'), '--checkpoint', str(a.checkpoint or source),
               '--data-root', str(root)]
        raise SystemExit(subprocess.call(cmd, cwd=CODE, env=env))
    runner, experiments = METHODS[a.method]
    if a.setting not in experiments:
        ap.error(f'{a.method} is prepared for the IRRG main setting only')
    experiment = experiments[a.setting]
    out = (a.output_dir/a.setting).resolve(); out.mkdir(parents=True, exist_ok=True)
    common = ['--workspace', str(CODE), '--settings', str(SETTINGS), '--data-root', str(root), '--source', str(source),
              '--output-root', str(out), '--seconds', str(a.hours*3600)]
    if runner == 'pheno':
        cmd = [sys.executable, '-u', str(CODE/'scripts/run_dtst_kaggle.py'), '--experiment', experiment,
               '--target-bands', a.setting.upper()] + common
    else:
        cmd = [sys.executable, '-u', str(CODE/'scripts/run_baseline_kaggle.py'), '--experiment', experiment] + common
        if experiment == 'snd_author':
            cmd += ['--warm', str(data/'snd_warm/seed1/warm'), '--warm-kind', 'snd_warm']
        if experiment == 'stcgn':
            cmd += ['--source-root', str(data/ROOTS['irrg'])]
    print(' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=CODE, env=env))


if __name__ == '__main__':
    main()
