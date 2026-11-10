"""Read frozen-subspace JSON artifacts and render paired scientific diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(root, name):
    return json.loads((root / name).read_text(encoding='utf-8'))


def miou(cm):
    cm = np.asarray(cm, dtype=np.int64)
    intersection = cm.diagonal()
    union = cm.sum(0) + cm.sum(1) - intersection
    return np.mean(intersection[union > 0] / union[union > 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    protocol, audit, geometry = [load(root, name) for name in ('PROTOCOL.json', 'BANK_AUDIT.json', 'GEOMETRY.json')]
    stage = 'full' if (root / 'METRICS_full.json').exists() else 'screen'
    result, decision = load(root, f'METRICS_{stage}.json'), load(root, 'DECISION.json')
    if len({protocol['signature'], audit['signature'], geometry['signature'], result['signature']}) != 1:
        raise ValueError('Artifacts do not belong to the same protocol')
    names = protocol['locked']['classes']
    routes = ['B0', 'B1', 'B2', 'M1', 'C1']
    scores = result['routes']
    source = scores['B0']['miou6']
    tile_delta = {tile: 100 * (miou(cm['M1']) - miou(cm['B0'])) for tile, cm in result['tile_confusion'].items()}
    positive_tiles = int(sum(v > 0 for v in tile_delta.values()))
    gains = {route: 100 * (scores['M1']['miou6'] - scores[route]['miou6']) for route in routes if route != 'M1'}
    changes = result['changes']['M1']
    worst = min((100 * (after - before), names[index]) for index, (before, after) in enumerate(
        zip(scores['B0']['iou'], scores['M1']['iou'])) if before is not None and after is not None)
    m1_pass = (gains['B0'] >= .5 and min(gains['B1'], gains['B2']) >= .3 and
               changes['wrong_to_correct'] > changes['correct_to_wrong'] and positive_tiles > 1)
    verdict = ('đạt ngưỡng thực dụng P2; còn cần P3' if stage == 'full' and m1_pass else
               'screen có tín hiệu dương; cần full evaluation' if stage == 'screen' and decision['p2_eligible'] else
               'chưa chứng minh ưu thế của subspace theo lớp với v1 đã khóa')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 150, 'savefig.dpi': 180})
    colors = ['#667085', '#2a9d8f', '#457b9d', '#e76f51', '#ad78b5']
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    values = [100 * scores[r]['miou6'] for r in routes]
    axes[0].bar(routes, values, color=colors)
    for index, value in enumerate(values):
        axes[0].text(index, value + .35, f'{value:.2f}', ha='center')
    axes[0].set_ylim(0, max(values) + 5)
    axes[0].set_ylabel('mIoU6 (%)')
    axes[0].set_title(f'{stage}: same checkpoint / same valid pixels')
    cls = np.arange(6)
    for index, route in enumerate(routes[1:]):
        delta = [100 * (a - b) if a is not None and b is not None else np.nan
                 for a, b in zip(scores[route]['iou'], scores['B0']['iou'])]
        axes[1].bar(cls + (index - 1.5) * .19, delta, width=.19, color=colors[index+1], label=route)
    axes[1].axhline(0, color='#444', linewidth=.8)
    axes[1].set_xticks(cls, [n.replace('_', '\n') for n in names], fontsize=8)
    axes[1].set_ylabel('IoU change vs B0 (pp)')
    axes[1].legend(ncol=4, fontsize=8)
    fig.savefig(figures / 'paired_metrics.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    counts = [geometry['classes'][str(i)]['sites'] for i in range(6)]
    axes[0].bar(cls, counts, color=['#2a9d8f' if geometry['classes'][str(i)]['supported'] else '#aaa' for i in range(6)])
    axes[0].axhline(256, color='#e76f51', linestyle='--', label='256-site minimum')
    axes[0].set_xticks(cls, [n.replace('_', '\n') for n in names], fontsize=8)
    axes[0].set_ylabel('Retained fit sites')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Green also passes >=3-tile support')
    for c in geometry['supported_classes']:
        spectrum = np.asarray(geometry['classes'][str(c)]['spectrum'])
        axes[1].plot(np.arange(1, min(64, len(spectrum)) + 1),
                     (np.cumsum(spectrum) / spectrum.sum())[:64], label=names[c])
    axes[1].axvline(8, color='#aaa', linestyle='--')
    axes[1].set_ylabel('Cumulative variance fraction')
    axes[1].set_xlabel('Number of eigen-directions')
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    overlaps = [(c, geometry['classes'][str(c)].get('tile_half_projection_overlap')) for c in geometry['supported_classes']]
    overlaps = [(c, v) for c, v in overlaps if v is not None]
    short_names = ['imperv.\nsurface', 'building', 'low\nvegetation', 'tree', 'car', 'clutter']
    axes[2].bar([short_names[c] for c, _ in overlaps], [v for _, v in overlaps], color='#457b9d')
    axes[2].tick_params(axis='x', labelsize=8)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('Mean squared cosine of principal angles')
    axes[2].set_title('PCA stability across disjoint tile groups')
    fig.savefig(figures / 'bank_geometry.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    tiles = sorted(tile_delta)
    ax.bar(tiles, [tile_delta[t] for t in tiles], color=['#2a9d8f' if tile_delta[t] > 0 else '#e76f51' for t in tiles])
    ax.axhline(0, color='#444', linewidth=.8)
    ax.set_ylabel('Within-tile mIoU6 change M1 vs B0 (pp)')
    ax.set_xlabel('Eval tile (these bars are not averaged to obtain the primary metric)')
    fig.savefig(figures / 'tile_deltas.png')
    plt.close(fig)

    lines = ['# Training-free subspace: kết quả kiểm chứng', '',
        f'Kết luận hiện tại: **{verdict}**.', '',
        'Caveat: checkpoint source được chọn bằng target validation ở step3000; source training dừng sớm trước budget40k. '
        'Đây là pilot exploratory trên một source seed, không phải independent test hay bằng chứng novelty.', '',
        '| Route | mIoU6 % | mIoU5 % | Δ B0 pp | Brier | ECE15 |', '|---|---:|---:|---:|---:|---:|']
    for route in routes:
        value = scores[route]
        lines.append(f"| {route} | {value['miou6']*100:.4f} | {value['miou5']*100:.4f} | {(value['miou6']-source)*100:+.4f} | {value['brier']:.4f} | {value['ece15']:.4f} |")
    lines += ['', 'B0 source-only; B1 isotropic prototype; B2 shared covariance; M1 class low-rank8; C1 random directions.', '',
              '| Class | B0 IoU % | M1 IoU % | Sites | Tiles | Supported |', '|---|---:|---:|---:|---:|---|']
    for index, name in enumerate(names):
        info = geometry['classes'][str(index)]
        before, after = scores['B0']['iou'][index], scores['M1']['iou'][index]
        display = lambda v: f'{v*100:.4f}' if v is not None else 'NA'
        lines.append(f"| {name} | {display(before)} | {display(after)} | {info['sites']} | {info['tiles']} | {info['supported']} |")
    corrected, damaged = changes['wrong_to_correct'], changes['correct_to_wrong']
    lines += ['', f'M1 sửa đúng {corrected:,} pixel; làm sai {damaged:,} pixel; net {corrected-damaged:+,}.',
        f"M1 − B1: {gains['B1']:+.4f}pp; M1 − B2: {gains['B2']:+.4f}pp; M1 − C1: {gains['C1']:+.4f}pp.",
        f"Tile tăng: {positive_tiles}/{len(tile_delta)}; worst class: {worst[1]} ({worst[0]:+.4f}pp).",
        f"Primary scope: {len(result['patches'])} {stage} patches, {scores['B0']['valid_pixels']:,} valid pixels/route.", '',
        f"Bank: {audit['forward_images']} forward-images; {audit['seconds']:.1f}s; feature storage {audit['feature_bytes']/1048576:.2f}MiB.",
        f"Fit + calibration: {geometry['fit_seconds']:.2f}s. Eval + diagnostics: {result['seconds']:.1f}s. Peak RSS: {result['peak_rss_mib']:.1f}MiB.",
        'Walltime includes diagnostics; per-route diagnostic time is not pure inference latency. CPU execution, no GPU memory claim.', '',
        '![Paired metrics](figures/paired_metrics.png)', '![Bank geometry](figures/bank_geometry.png)', '![Tile deltas](figures/tile_deltas.png)', '']
    boundary_pixels = result['boundary']['B0']['valid_pixels']
    lines += [f'Boundary diagnostic valid pixels: {boundary_pixels:,}. ' +
        ('Native masks eroded: no valid 4-neighbor class transitions in this scope; boundary metrics are null. No boundary-quality conclusion.'
         if boundary_pixels == 0 else 'Boundary means both valid sides of 4-neighbor GT transitions; width1.'), '',
        'Giữ nguyên patch aggregation của native evaluator: edge patches overlap; không tuyên bố full-tile stitching. '
        'mIoU5 vẫn tính từ confusion6x6, chỉ bỏ clutter khỏi trung bình.', '',
        'P0 parity và weight/buffer invariance đã PASS. Target-train labels không được đọc; calibration tiles không vào fit bank. '
        'Sampling/gate/rank/alpha giữ nguyên sau khi xem kết quả.', '',
        'P3 official comparators, seed2/3, sensitivity và P4 G2C còn có điều kiện. Semantic-margin/gate-quality proxy và correction maps '
        'chưa được triển khai trong pilot; không thay bằng pseudo-label agreement hay entropy để kết luận accuracy.', '',
        'Development history: attempt1 lỗi JSON vì empty region CM; sửa null handling rồi rerun cùng protocol ở r2. '
        'Không chọn tham số theo điểm attempt1.']
    if stage == 'full':
        lines += ['', 'Paired bootstrap2000 theo nguyên tile:']
        for route, data in result['paired_tile_bootstrap_2000'].items():
            lo, hi = data['delta_vs_B0_pp_ci95']
            lines.append(f'- {route}−B0: CI95% [{lo:+.4f}, {hi:+.4f}]pp.')
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    reporting = {'protocol_signature': protocol['signature'], 'stage': stage,
        'script_sha256': sha(Path(__file__)), 'raw_metrics_sha256': sha(root / f'METRICS_{stage}.json'),
        'files_sha256': {str(path.relative_to(root)): sha(path) for path in
                        [root / 'REPORT.md', *sorted(figures.glob('*.png'))]}}
    (root / 'REPORTING.json').write_text(json.dumps(reporting, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'stage': stage, 'verdict': verdict, 'm1_deltas_pp': gains,
                      'positive_tiles': positive_tiles, 'figures': 3}))


if __name__ == '__main__':
    main()
