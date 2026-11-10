"""Paired B2 latent-aux reports from portable ZIPs; no model/data access."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import zipfile


def read_json(archive, name):
    return json.loads(archive.read(name))


def read_curve(archive, name):
    return [json.loads(line) for line in archive.read(name).decode().splitlines() if line]


def summarize(baseline_zip, run_zips):
    with zipfile.ZipFile(baseline_zip) as archive:
        baseline_protocol = read_json(archive, 'protocol.json')
        baseline_result = read_json(archive, 'result_portable.json')
        baseline_warm = read_json(archive, 'warm_selection.json')
        baseline_curve = read_curve(archive, 'snd/evaluation_curve.jsonl')
    baseline = {point['iteration']: point for point in baseline_curve}
    baseline_recipe = {'readout_lr_mult': 10, 'freeze_source_readout': False, **baseline_protocol['recipe']}
    source_sha = baseline_protocol['source_checkpoint_sha256']
    rows, curves = [], []
    for path in run_zips:
        with zipfile.ZipFile(path) as archive:
            protocol = read_json(archive, 'protocol.json')
            result = read_json(archive, 'result_portable.json')
            points = read_curve(archive, 'snd/evaluation_curve.jsonl')
        aux_start = protocol.get('aux_start', 'snd')
        recipe = {'readout_lr_mult': 10, 'freeze_source_readout': False, **protocol['recipe']}
        if (protocol['method'] != 'snd_srmp_aux' or protocol['source_checkpoint_sha256'] != source_sha or
                protocol['seed'] != baseline_protocol['seed'] or
                protocol['manifests'] != baseline_protocol['manifests'] or
                recipe != baseline_recipe or
                protocol['warm_protocol_signature'] != result.get('warm_protocol_signature', protocol['warm_protocol_signature'])):
            raise ValueError(f'{path}: source, data or recipe mismatch')
        if aux_start == 'warm' and 'warm_protocol_signature' not in result:
            raise ValueError(f'{path}: warm-and-SND handoff provenance is missing')
        if aux_start == 'snd':
            if (protocol['warm_protocol_signature'] != baseline_protocol['protocol_signature'] or
                    protocol['warm_checkpoint_sha256'] != baseline_warm['checkpoint_sha256']):
                raise ValueError(f'{path}: selected baseline warm mismatch')
        elif aux_start != 'warm':
            raise ValueError(f'{path}: unknown auxiliary start')
        if protocol['warm_checkpoint_sha256'] != result['warm_selection']['checkpoint_sha256']:
            raise ValueError(f'{path}: selected warm checkpoint mismatch')
        objective = protocol['auxiliary']['objective']
        aux_weight = protocol['auxiliary']['weight']
        warm_early = result.get('warm_stopped_early', False)
        for point in points:
            curves.append({'seed': protocol['seed'], 'objective': objective, 'aux_start': aux_start,
                           'aux_weight': aux_weight, 'iteration': point['iteration'],
                           'miou6': point['metrics']['miou'], 'miou5': point['metrics']['miou5']})
        common = [(baseline[point['iteration']], point) for point in points
                  if point['iteration'] in baseline and point['iteration'] > 0]
        if not common:
            raise ValueError(f'{path}: no matching continuation evaluation steps')
        base_last, aux_last = common[-1]
        base_peak = max((pair[0] for pair in common), key=lambda point: point['metrics']['miou'])
        aux_peak = max((pair[1] for pair in common), key=lambda point: point['metrics']['miou'])
        full_pair = (baseline_curve[-1]['iteration'] == protocol['recipe']['snd']['steps'] and
                     result['continuation_complete'] and points[-1]['iteration'] == protocol['recipe']['snd']['steps'] and
                     not warm_early)
        tail = [pair for pair in common if pair[1]['iteration'] >= .8 * protocol['recipe']['snd']['steps']]
        rows.append({'seed': protocol['seed'], 'objective': objective, 'aux_start': aux_start,
                     'aux_weight': aux_weight,
                     'comparison_scope': ('same_selected_warm' if aux_start == 'snd' else
                                          'different_warm_early_stopped_pipeline' if warm_early else
                                          'different_warm_full_pipeline'),
                     'warm_actual_updates': result.get('warm_actual_updates'),
                     'warm_planned_updates': result.get('warm_planned_updates'),
                     'source_sha256': source_sha, 'warm_sha256': protocol['warm_checkpoint_sha256'],
                     'warm_final_delta_pp': 100 * (result['warm_final']['metrics']['miou'] -
                                                   baseline_result['results']['warm']['metrics']['miou']),
                     'baseline_last_iteration': baseline_curve[-1]['iteration'],
                     'aux_last_iteration': points[-1]['iteration'], 'paired_last_iteration': aux_last['iteration'],
                     'paired_last_delta_pp': 100 * (aux_last['metrics']['miou'] - base_last['metrics']['miou']),
                     'paired_last_miou5_delta_pp': 100 * (aux_last['metrics']['miou5'] - base_last['metrics']['miou5']),
                     'paired_peak_delta_pp': 100 * (aux_peak['metrics']['miou'] - base_peak['metrics']['miou']),
                     'paired_last_iou_delta_pp': {
                         name: 100 * (aux_last['metrics']['iou_by_class'][name] - base_last['metrics']['iou_by_class'][name])
                         for name in base_last['metrics']['class_order']},
                     'aux_last_prediction_fraction': aux_last['metrics']['prediction_fraction'],
                     'paired_tail_delta_pp': (100 * statistics.mean(a['metrics']['miou'] - b['metrics']['miou']
                                                                  for b, a in tail) if full_pair and tail else None),
                     'full_budget_paired': full_pair,
                     'final_delta_pp': (100 * (points[-1]['metrics']['miou'] - baseline_curve[-1]['metrics']['miou'])
                                        if full_pair else None),
                     'aux_peak_iteration': max(points, key=lambda point: point['metrics']['miou'])['iteration'],
                     'aux_peak_miou6': max(point['metrics']['miou'] for point in points),
                     'aux_peak_miou5': max(point['metrics']['miou5'] for point in points),
                     'aux_peak_miou5_iteration': max(points, key=lambda point: point['metrics']['miou5'])['iteration'],
                     'aux_post_start_peak_miou6': max(point['metrics']['miou'] for point in points
                                                     if point['iteration'] > 0),
                     'baseline_status': ('full' if baseline_curve[-1]['iteration'] == protocol['recipe']['snd']['steps']
                                         else 'early_stopped')})
    return {'protocol': 'siamseg_sfuda_b2_v1', 'baseline_source_sha256': source_sha,
            'selection_note': 'PD14 target validation selects peaks; it is not an independent test',
            'runs': rows, 'curve': curves}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-zip', type=Path, required=True)
    parser.add_argument('--run-zip', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.baseline_zip, args.run_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({k: v for k, v in report.items() if k != 'curve'}, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('seed', 'objective', 'aux_start', 'aux_weight',
                                                    'iteration', 'miou6', 'miou5'))
        writer.writeheader(); writer.writerows(report['curve'])
    print(args.output)


if __name__ == '__main__':
    main()
