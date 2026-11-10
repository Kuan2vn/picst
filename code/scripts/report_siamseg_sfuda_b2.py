"""Summarize portable SND B2 runs without reading model weights or target masks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def curve(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.is_file() else []


def summarize(root):
    source = sorted((root/'source').glob('seed*/selected_target_val_step*.pth'))
    if not source:
        source = sorted((root/'source').glob('seed*/best_target_val.pth'))
    if len(source) != 1:
        raise ValueError('Report requires exactly one fixed source checkpoint')
    source_sha = sha256(source[0])
    seeds, curves = [], []
    for directory in sorted((root/'snd').glob('seed*')):
        protocol_path = directory/'protocol.json'
        if not protocol_path.is_file():
            continue
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        if protocol.get('source_checkpoint_sha256') != source_sha or protocol.get('protocol') != 'siamseg_sfuda_b2_v1':
            raise ValueError(f'{directory}: source/protocol mismatch')
        result_path = directory/'result_portable.json'
        if not result_path.is_file():
            seeds.append({'seed': protocol['seed'], 'status': 'incomplete',
                          'error': json.loads((directory/'error.json').read_text()) if (directory/'error.json').is_file() else None})
            continue
        final = json.loads(result_path.read_text(encoding='utf-8'))['results']
        if not {'source', 'warm', 'snd', 'meta-off'} <= set(final):
            seeds.append({'seed': protocol['seed'], 'status': 'incomplete'})
            continue
        baseline = final['source']['metrics']
        count = baseline['valid_pixels']
        if any(item['metrics']['valid_pixels'] != count for item in final.values()):
            raise ValueError(f'{directory}: evaluation pixel counts differ')
        warm = curve(directory/'warm/evaluation_curve.jsonl')
        snd = curve(directory/'snd/evaluation_curve.jsonl')
        off = curve(directory/'meta-off/evaluation_curve.jsonl')
        if not warm or not snd or not off:
            raise ValueError(f'{directory}: missing evaluation curves')
        selection_path = directory/'warm_selection.json'
        warm_iteration = (json.loads(selection_path.read_text())['iteration'] if selection_path.is_file()
                          else protocol['recipe']['warm']['steps'])
        selected_warm = next((point for point in warm if point['iteration'] == warm_iteration), None)
        if selected_warm is None:
            raise ValueError(f'{directory}: selected warm iteration missing from curve')
        for method, points in (('warm', warm), ('snd', snd), ('meta-off', off)):
            for point in points:
                curves.append({'seed': protocol['seed'], 'method': method,
                               'iteration': point['total_iteration'],
                               'miou6': point['metrics']['miou'],
                               'valid_pixels': point['metrics']['valid_pixels']})
        best_meta = max(snd, key=lambda row: row['metrics']['miou'])
        best_off = max(off, key=lambda row: row['metrics']['miou'])
        best_pipeline = max([point for point in warm if point['iteration'] <= warm_iteration] + snd,
                            key=lambda row: row['metrics']['miou'])
        budget = protocol['recipe']['warm']['steps'] + protocol['recipe']['snd']['steps']
        tail = [point['metrics']['miou'] for point in snd if point['total_iteration'] >= .8*budget]
        peak_file = directory/'snd/best_target_val.pth'
        source_miou = baseline['miou']
        snd_final = final['snd']['metrics']
        off_final = final['meta-off']['metrics']
        seeds.append({
            'seed': protocol['seed'], 'status': 'complete',
            'source_sha256': source_sha, 'source_miou6': source_miou,
            'warm_final_miou6': final['warm']['metrics']['miou'],
            'warm_selected_iteration': warm_iteration,
            'warm_selected_miou6': selected_warm['metrics']['miou'],
            'snd_peak_miou6': best_meta['metrics']['miou'],
            'meta_off_peak_miou6': best_off['metrics']['miou'],
            'snd_final_miou6': snd_final['miou'], 'meta_off_final_miou6': off_final['miou'],
            'gain_final_pp': 100*(snd_final['miou']-source_miou),
            'gain_peak_meta_pp': 100*(best_meta['metrics']['miou']-source_miou),
            'gain_peak_vs_meta_off_peak_pp': 100*(best_meta['metrics']['miou']-best_off['metrics']['miou']),
            'gain_peak_pipeline_pp': 100*(best_pipeline['metrics']['miou']-source_miou),
            'peak_meta_iteration': best_meta['total_iteration'],
            'peak_pipeline_iteration': best_pipeline['total_iteration'],
            'peak_pipeline_stage': best_pipeline['stage'],
            'peak_meta_checkpoint_sha256': sha256(peak_file) if peak_file.is_file() else None,
            'gain_vs_warm_pp': 100*(snd_final['miou']-final['warm']['metrics']['miou']),
            'gain_vs_warm_selected_pp': 100*(snd_final['miou']-selected_warm['metrics']['miou']),
            'gain_vs_meta_off_pp': 100*(snd_final['miou']-off_final['miou']),
            'gain_vs_meta_off_by_class_pp': {
                name: (100*(value-off_final['iou_by_class'][name])
                       if value is not None and off_final['iou_by_class'][name] is not None else None)
                for name, value in snd_final['iou_by_class'].items()},
            'peak_to_final_drop_pp': 100*(best_pipeline['metrics']['miou']-snd_final['miou']),
            'tail_gain_pp': 100*(statistics.mean(tail)-source_miou) if tail else None,
            'tail_min_miou6': min(tail) if tail else None,
            'tail_max_miou6': max(tail) if tail else None,
            'tail_above_source_fraction': sum(value > source_miou for value in tail)/len(tail) if tail else None,
            'tail_eval_count': len(tail),
            'valid_pixels': count,
        })
    complete = [entry for entry in seeds if entry['status'] == 'complete']
    aggregates = {}
    for field in ('gain_final_pp', 'gain_peak_meta_pp', 'gain_peak_vs_meta_off_peak_pp',
                  'gain_peak_pipeline_pp',
                  'gain_vs_warm_pp', 'gain_vs_warm_selected_pp', 'gain_vs_meta_off_pp',
                  'peak_to_final_drop_pp', 'tail_gain_pp'):
        values = [entry[field] for entry in complete if entry[field] is not None]
        aggregates[field] = {'count': len(values), 'mean': statistics.mean(values) if values else None,
                             'sample_std': statistics.stdev(values) if len(values) > 1 else None,
                             'min': min(values) if values else None, 'max': max(values) if values else None,
                             'positive_seeds': sum(value > 0 for value in values)}
    return {'protocol': 'siamseg_sfuda_b2_v1', 'source_checkpoint_sha256': source_sha,
            'selection': 'target_validation_selected; same PD14 is not an independent test',
            'seeds': seeds, 'aggregate': aggregates, 'curve': curves}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    output = args.root/'snd_gain_report.json'
    output.write_text(json.dumps({key: value for key, value in report.items() if key != 'curve'}, indent=2)+'\n', encoding='utf-8')
    with (args.root/'snd_gain_curve.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('seed', 'method', 'iteration', 'miou6', 'valid_pixels'))
        writer.writeheader(); writer.writerows(report['curve'])
    portable = args.root/'snd_gain_portable.zip'
    with zipfile.ZipFile(portable, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.root.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.log', '.md'):
                archive.write(path, path.relative_to(args.root))
    print(output)


if __name__ == '__main__':
    main()
