"""Report target-validation peak and the verified last result, including early stops."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile


def summarize(path):
    path = Path(path)
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError('Corrupt report archive')
            names = ('protocol.json', 'warm/complete.json', 'warm/stopped_early.json',
                     'warm/evaluation_curve.jsonl', 'warm/observed_peak.json', 'early_stopping_policy.json',
                     'warm/user_stop_override.json', 'pilot_budget.json')
            files = {name: archive.read(name).decode('utf-8-sig') for name in names if name in archive.namelist()}
    else:
        names = ('protocol.json', 'warm/complete.json', 'warm/stopped_early.json',
                 'warm/evaluation_curve.jsonl', 'warm/observed_peak.json', 'early_stopping_policy.json',
                 'warm/user_stop_override.json', 'pilot_budget.json')
        files = {name: (path/name).read_text(encoding='utf-8-sig') for name in names if (path/name).exists()}
    protocol = json.loads(files['protocol.json'])
    early = 'warm/stopped_early.json' in files
    if early and 'warm/complete.json' in files:
        raise ValueError('Run has both early-stop and full-budget completion markers')
    complete = json.loads(files['warm/stopped_early.json' if early else 'warm/complete.json'])
    policy = json.loads(files['early_stopping_policy.json']) if 'early_stopping_policy.json' in files else None
    curve = [json.loads(line) for line in files['warm/evaluation_curve.jsonl'].splitlines() if line.strip()]
    signature = protocol['protocol_signature']
    steps = [point['iteration'] for point in curve]
    if (not curve or steps != sorted(set(steps)) or complete['protocol_signature'] != signature or
            any(not math.isfinite(point['metrics']['miou']) or not 0 <= point['metrics']['miou'] <= 1 or
                point['protocol_signature'] != signature or point['model'] != 'EMA' or
                point['image_count'] != curve[0]['image_count'] for point in curve)):
        raise ValueError('Evaluation provenance/order mismatch')
    endpoint = next(point for point in curve if point['iteration'] == complete['iteration'])
    planned = protocol['recipe']['warm']['steps']
    if not early and complete['iteration'] != planned:
        raise ValueError('Run has not completed the fixed adaptation budget')
    period = protocol['recipe'].get('eval_period', steps[1] if len(steps)>1 else 3000)
    expected = list(range(0,planned+1,period))
    if expected[-1] != planned:
        expected.append(planned)
    if steps != [step for step in expected if step <= complete['iteration']]:
        raise ValueError('Evaluation cadence differs from the declared protocol')
    manual = early and complete.get('user_requested_stop') is True
    override = None
    effective_policy = policy
    pilot = json.loads(files['pilot_budget.json']) if 'pilot_budget.json' in files else None
    pilot_stop = early and complete.get('pilot_budget_cap') is True
    if pilot:
        try:
            from scripts.dtst_pilot_budget import validate_budget, cap_extension_active, cap_decision, effective_policy as pilot_policy
        except ModuleNotFoundError:
            from dtst_pilot_budget import validate_budget, cap_extension_active, cap_decision, effective_policy as pilot_policy
        cap = validate_budget(pilot, policy, protocol)
        extended = cap_extension_active(curve, pilot, policy)
        limit = policy['max_updates'] if extended else cap
        if complete['iteration'] > limit or manual or (not early and not extended):
            raise ValueError('Run exceeds or conflicts with its declared pilot budget')
        effective_policy = pilot_policy(policy, pilot)
    elif pilot_stop:
        raise ValueError('Pilot stop has no declared pilot budget')
    extension_stop = early and complete.get('pilot_cap_extension') is True
    if extension_stop:
        decision = cap_decision(curve, pilot, policy) if pilot else None
        earlier_stop = any(cap_decision(curve[:i], pilot, policy) is not None
                           for i in range(1,len(curve))) if pilot else True
        if (not decision or earlier_stop or complete.get('pilot_budget') != pilot or
                decision['iteration'] != complete['iteration'] or
                complete['planned_iteration'] != planned or complete['stopping_policy'] != policy):
            raise ValueError('Invalid first-decline cap extension marker')
    elif pilot_stop:
        if (complete.get('pilot_budget') != pilot or complete['iteration'] != cap or
                complete['planned_iteration'] != planned or complete['stopping_policy'] != policy):
            raise ValueError('Invalid pilot budget completion marker')
    elif manual:
        override = json.loads(files.get('warm/user_stop_override.json', '{}'))
        cap = override.get('requested_stop_update')
        effective_policy = dict(policy or {}, user_requested_budget_cap=cap, posthoc_budget_override=True)
        if (protocol['gsd_study']['arm'] != 'generic' or cap != 12000 or
                override.get('original_policy') != policy or not override.get('posthoc_budget_override') or
                not policy or policy['max_updates'] != planned or
                complete['planned_iteration'] != planned or complete['iteration'] != cap or
                complete['stopping_policy'] != effective_policy):
            raise ValueError('Invalid user-requested generic stop override')
    elif early:
        try:
            from scripts.dtst_run_control import stop_decision
        except ModuleNotFoundError:
            from dtst_run_control import stop_decision
        if (not policy or complete['stopping_policy'] != policy or policy['max_updates'] != planned or
                complete['planned_iteration'] != planned or stop_decision(curve, policy) is None or
                stop_decision(curve, policy)['iteration'] != complete['iteration']):
            raise ValueError('Invalid early-stop decision/policy')
    peak = max(curve, key=lambda point: point['metrics']['miou'])
    observed = json.loads(files['warm/observed_peak.json'])
    if observed['iteration'] != peak['iteration'] or observed['metrics'] != peak['metrics']:
        raise ValueError('Saved peak differs from the evaluation curve')
    if (pilot_stop or extension_stop) and (complete.get('peak_iteration') != peak['iteration'] or
                       complete.get('peak_miou') != peak['metrics']['miou']):
        raise ValueError('Pilot stop peak metadata differs from the evaluation curve')
    result = dict(arm=protocol['gsd_study']['arm'], seed=protocol['seed'],
                  loss_mode=protocol['gsd_study'].get('loss_mode'),
                  protocol_signature=signature, recipe=protocol['recipe'],
                  source_checkpoint_sha256=protocol['source_checkpoint_sha256'],
                  manifests=protocol['manifests'], image_count=peak['image_count'],
                  evaluation_updates=steps, selection='target_validation_selected',
                  target_labels_used_for_gradients=False,
                  target_labels_used_for_checkpoint_selection=True,
                  target_labels_used_for_stopping=policy is not None,
                  stopping_policy=effective_policy, planned_updates=planned, actual_updates=complete['iteration'],
                  declared_stopping_policy=policy, user_stop_override=override,
                  pilot_budget=pilot, pilot_max_updates=pilot['max_updates'] if pilot else None,
                  stopping=('pilot_extension_first_decline' if extension_stop else 'pilot_budget_cap' if pilot_stop else 'user_requested_budget_cap' if manual else 'target_validation_early_stopped') if early else ('max_budget_reached' if policy else 'fixed_budget'),
                  source_metrics=curve[0]['metrics'], runtime=complete,
                  peak=dict(iteration=peak['iteration'], metrics=peak['metrics'],
                            checkpoint='warm/best_target_val.pth'),
                  endpoint=dict(iteration=endpoint['iteration'], metrics=endpoint['metrics'],
                                checkpoint='warm/last.pth', is_fixed_budget_endpoint=not early))
    if path.is_dir():
        # CPU-only verification of our own saved checkpoints, with no new GPU evaluation.
        import torch
        for kind in ('peak', 'endpoint'):
            checkpoint = path/result[kind]['checkpoint']
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['iteration'] != result[kind]['iteration'] or state['protocol_signature'] != signature:
                raise ValueError('Checkpoint iteration/signature differs from reported metrics')
            with checkpoint.open('rb') as stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    hasher.update(chunk)
                digest = hasher.hexdigest()
            if kind == 'endpoint' and digest != complete['checkpoint_sha256']:
                raise ValueError('Endpoint checkpoint checksum mismatch')
            result[kind].update(checkpoint_sha256=digest, checkpoint_metadata_verified=True)
            del state
    return result, curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+', help='Completed run directories or portable report ZIPs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-continuation', action='store_true', help='Report only inside a registered parent queue')
    args = parser.parse_args()
    if len(args.reports)==1 and args.reports[0].is_dir():
        try:
            from scripts.dtst_deferred_run import deferred_run
        except ModuleNotFoundError:
            from dtst_deferred_run import deferred_run
        declaration=deferred_run(args.reports[0])
        if declaration:
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'DEFERRED.json').write_text(json.dumps(dict(status='deferred',
                actual_updates=0,completed=False,peak=None,declaration=declaration),indent=2)+'\n')
            (args.output/'DEFERRED.md').write_text('Run deferred before training to prioritize method pilots and conditional controls. No model, peak or endpoint exists.\n')
            print(json.dumps(dict(status='deferred',phase=declaration['phase'],actual_updates=0)))
            return
    summaries, curves = {}, {}
    for path in args.reports:
        report, curve = summarize(path)
        mode = report['loss_mode']
        variant = report['arm'] + ('_'+mode if mode else '')
        key = f"{variant}_seed{report['seed']}"
        if key in summaries:
            raise ValueError('Duplicate arm/seed')
        summaries[key], curves[key] = report, curve
    first = next(iter(summaries.values()))
    if any(report[key] != first[key] for report in summaries.values() for key in
           ('source_checkpoint_sha256', 'manifests', 'recipe', 'stopping_policy', 'image_count')):
        raise ValueError('Compared arms have different evaluation/training protocols')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'peak_and_endpoint.json').write_text(json.dumps(summaries, indent=2)+'\n', encoding='utf-8')
    lines = ['# DT-ST: peak and last measured result', '',
             '| Arm | Peak mIoU6 (%) | Peak update | Last mIoU6 (%) | Last update | Stop reason |',
             '|---|---:|---:|---:|---:|---|']
    for key, report in summaries.items():
        peak, endpoint = report['peak'], report['endpoint']
        lines.append(f"| {key} | {100*peak['metrics']['miou']:.4f} | {peak['iteration']} | {100*endpoint['metrics']['miou']:.4f} | {endpoint['iteration']} | {report['stopping']} |")
    lines += ['', 'Peak uses target evaluation labels for checkpoint selection. All secondary/per-class metrics come from that same mIoU6-selected checkpoint.',
              'Early stopping, when declared, uses target validation labels; actual budgets can differ under the same stopping rule.',
              'An early-stopped last result is not a20k endpoint. Peak is the maximum among recorded evaluations.', '']
    if any(report['user_stop_override'] for report in summaries.values()):
        lines.append('User-requested caps are posthoc overrides, not numerical early-stop decisions; do not treat them as a matched stopping protocol.')
    if any(report['pilot_budget'] for report in summaries.values()):
        lines.append('Pilot caps do not establish convergence or a20k endpoint. Compare different caps only within the same observed update window; LR horizon remains32000.')
    (args.output/'PEAK_AND_ENDPOINT.md').write_text('\n'.join(lines), encoding='utf-8')
    with (args.output/'evaluation_curve.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        classes = first['source_metrics']['class_order']
        writer.writerow(['arm_seed', 'iteration', 'miou6_percent', 'car_precision_percent',
                         'car_recall_percent', *[f'{name}_iou_percent' for name in classes]])
        for key, curve in curves.items():
            for point in curve:
                metrics = point['metrics']
                writer.writerow([key, point['iteration'], 100*metrics['miou'],
                                 100*metrics['precision_by_class']['car'], 100*metrics['recall_by_class']['car'],
                                 *[100*metrics['iou_by_class'][name] for name in classes]])
    print(json.dumps({key: {'peak_miou6': 100*r['peak']['metrics']['miou'],
                            'peak_iteration': r['peak']['iteration'],
                            'endpoint_miou6': 100*r['endpoint']['metrics']['miou']}
                      for key, r in summaries.items()}))
    if len(args.reports)==1 and not args.skip_continuation:
        try:
            from scripts.dtst_scout_queue import after_training
        except ModuleNotFoundError as error:
            if error.name != 'scripts':
                raise
            from dtst_scout_queue import after_training
        after_training(args.reports[0])


if __name__ == '__main__':
    main()
