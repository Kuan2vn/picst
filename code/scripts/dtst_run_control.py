"""Validation early stopping for our own DT-ST processes in an existing allocation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read_curve(path):
    records = []
    if path.exists():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines)-1:
                    raise
    return records


def stop_decision(curve, policy):
    if not curve or curve[-1]['iteration'] < policy['min_updates'] or curve[-1]['iteration'] >= policy['max_updates']:
        return None
    patience = policy['patience_evaluations']
    if len(curve) <= patience:
        return None
    peak = max(curve, key=lambda point: point['metrics']['miou'])
    recent = curve[-patience:]
    values = [point['metrics']['miou'] for point in recent]
    if policy.get('mode') == 'no_improvement':
        # Equal scores do not reset patience; the first occurrence owns a peak.
        if all(point['iteration'] > peak['iteration'] for point in recent):
            return dict(iteration=recent[-1]['iteration'], peak_iteration=peak['iteration'],
                        peak_miou=peak['metrics']['miou'], recent_miou=values,
                        reason='Two consecutive validations without a strictly higher mIoU6 peak')
        return None
    requires_decline = policy.get('require_continuing_decline', True)
    below_peak = all(peak['metrics']['miou']-value >= policy['below_peak_margin'] for value in values)
    continuing_decline = all(a-b >= policy['consecutive_decline_margin'] for a, b in zip(values, values[1:]))
    if below_peak and (not requires_decline or continuing_decline):
        return dict(iteration=recent[-1]['iteration'], peak_iteration=peak['iteration'],
                    peak_miou=peak['metrics']['miou'], recent_miou=values,
                    reason=('Consecutive evaluations below peak with continuing decline' if requires_decline else
                            'Consecutive evaluations below peak by the declared margin'))
    return None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def process_running(pid):
    # After SIGTERM, /proc/environ can become unreadable before the PID disappears.
    try:
        state=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]
        return state not in ('Z','X')
    except FileNotFoundError:
        return False


def checked_process(pid, run, allocation):
    proc = Path('/proc')/str(pid)
    try:
        if proc.stat().st_uid != os.getuid():
            raise RuntimeError('Refusing to control a process owned by someone else')
        command = [part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
        if not command:
            return False
        environment = (proc/'environ').read_bytes().split(b'\0')
        identity = ('DTST_LOCAL_RUN_ID='+allocation[6:] if allocation.startswith('local:')
                    else 'SLURM_JOB_ID='+allocation)
        if identity.encode() not in environment:
            raise RuntimeError('Trainer is outside the authorized allocation')
        entry = next((Path(part).name for part in command[1:] if part.endswith('.py')),None)
        if entry != 'train_DTST_remote.py':
            raise RuntimeError('Not the expected DT-ST trainer')
        if (proc/'comm').read_text().strip() == 'pt_data_worker':
            raise RuntimeError('Refusing to signal a DataLoader worker')
        ppid = (proc/'stat').read_text().split(') ',1)[1].split()[1]
        parent = Path('/proc')/ppid/'cmdline'
        if parent.exists():
            parent_command = [part.decode() for part in parent.read_bytes().split(b'\0') if part]
            parent_entry = next((Path(part).name for part in parent_command[1:] if part.endswith('.py')),None)
            if parent_entry == 'train_DTST_remote.py':
                raise RuntimeError('Refusing to signal a forked trainer worker')
        if '--output' not in command or Path(command[command.index('--output')+1]).resolve() != run.resolve():
            raise RuntimeError('Trainer output differs from the intended run')
        return True
    except FileNotFoundError:
        return False
    except PermissionError:
        if not process_running(pid):
            return False
        raise


def stop_at_checkpoint(pid, run, allocation, decision, policy, stop_kind='early_stop'):
    if stop_kind not in ('early_stop', 'session_pause'):
        raise ValueError('Unknown checkpoint stop kind')
    import torch
    stage = run/'warm'
    if (stage/'complete.json').exists() or (stop_kind=='session_pause' and decision['iteration']>=policy['max_updates']):
        return False
    if not checked_process(pid, run, allocation):
        return False
    os.kill(pid, signal.SIGSTOP)
    terminated = False
    try:
        if any(stage.glob('*.pth.tmp')):
            return False
        protocol = json.loads((run/'protocol.json').read_text())
        last = torch.load(stage/'last.pth', map_location='cpu', weights_only=False)
        resume = torch.load(stage/'resume.pth', map_location='cpu', weights_only=False)
        peak = torch.load(stage/'best_target_val.pth', map_location='cpu', weights_only=False)
        if last['iteration'] != decision['iteration'] or resume['iteration'] != decision['iteration']:
            return False
        for state in (last, resume, peak):
            if (state['protocol_signature'] != protocol['protocol_signature'] or
                    state['source_checkpoint_sha256'] != protocol['source_checkpoint_sha256']):
                raise ValueError('Early-stop checkpoint provenance mismatch')
        if peak['iteration'] != decision['peak_iteration']:
            return False
        latest = read_curve(stage/'training_history.jsonl')
        record = dict(decision, protocol_signature=protocol['protocol_signature'],
                      planned_iteration=policy['max_updates'], stopping_policy=policy,
                      checkpoint_sha256=digest(stage/'last.pth'),
                      resume_checkpoint_sha256=digest(stage/'resume.pth'),
                      peak_checkpoint_sha256=digest(stage/'best_target_val.pth'),
                      last_logged_iteration=latest[-1]['iteration'] if latest else decision['iteration'],
                      retained_iteration=decision['iteration'],
                      target_labels_used_for_stopping=stop_kind=='early_stop',
                      stopped_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(),
                      complete_fixed_budget=False)
        # Only terminate this verified trainer; its allocation and interactive shell remain live.
        if not checked_process(pid, run, allocation):
            return False
        write_json(stage/'early_stop_intent.json',dict(record,trainer_pid=pid,allocation=allocation))
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        terminated = True
        for _ in range(60):
            if not process_running(pid):
                break
            time.sleep(.5)
        else:
            raise RuntimeError('Trainer has not exited; no early-stop marker written')
        write_json(stage/('stopped_early.json' if stop_kind=='early_stop' else 'session_pause.json'), record)
        label='Early stopped' if stop_kind=='early_stop' else 'Session paused'
        print(f'{label} {run} at retained checkpoint{decision["iteration"]}; peak{decision["peak_iteration"]}.', flush=True)
        return True
    finally:
        if not terminated and checked_process(pid, run, allocation):
            os.kill(pid, signal.SIGCONT)


def monitor(pid, run, allocation, policy):
    run.mkdir(parents=True, exist_ok=True)
    sidecar = run/'early_stopping_policy.json'
    if sidecar.exists() and json.loads(sidecar.read_text()) != policy:
        raise ValueError('Run already has a different stopping policy')
    write_json(sidecar, policy)
    stage, observed = run/'warm', None
    while checked_process(pid, run, allocation):
        if (stage/'complete.json').exists():
            return
        curve = read_curve(stage/'evaluation_curve.jsonl')
        if curve and observed != curve[-1]['iteration']:
            observed = curve[-1]['iteration']
            print(f'Early-stop monitor {run.name}: eval{observed}, mIoU{100*curve[-1]["metrics"]["miou"]:.4f}%.', flush=True)
        decision = stop_decision(curve, policy)
        budget_path = run/'pilot_budget.json'
        if budget_path.exists():
            try:
                from scripts.dtst_pilot_budget import cap_decision, cap_extension_active, validate_budget
            except ModuleNotFoundError:
                from dtst_pilot_budget import cap_decision, cap_extension_active, validate_budget
            budget = json.loads(budget_path.read_text())
            if (run/'protocol.json').exists():
                validate_budget(budget, policy, json.loads((run/'protocol.json').read_text()))
            cap_stop = cap_decision(curve, budget, policy)
            decision = cap_stop if cap_extension_active(curve, budget, policy) else decision or cap_stop
        if decision and stop_at_checkpoint(pid, run, allocation, decision, policy):
            return
        time.sleep(5)


def adopt_pair(root, state_path, allocation, policy):
    for arm in ('raw', 'gsd'):
        run = root/arm/'seed1'
        run.mkdir(parents=True, exist_ok=True)
        write_json(run/'early_stopping_policy.json', policy)
    while True:
        state = json.loads(state_path.read_text())
        if state['allocation'] != allocation or state['local_results'] != str(root):
            raise ValueError('Existing pair is not the intended allocation/output')
        if state['status'] in ('complete', 'failed'):
            if state['status'] == 'failed' and not any((root/arm/'seed1/warm/stopped_early.json').exists() for arm in ('raw','gsd')):
                raise RuntimeError('Original pair failed without an approved early-stop checkpoint; inspect before continuing')
            print(f'Original pair launcher exited ({state["status"]}); managed continuation will verify retained results.', flush=True)
            return
        found = False
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if proc.stat().st_uid != os.getuid():
                    continue
                command = [part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
                entry = next((Path(part).name for part in command[1:] if part.endswith('.py')),None)
                if '--output' not in command or entry != 'train_DTST_remote.py' or (proc/'comm').read_text().strip() == 'pt_data_worker':
                    continue
                run = Path(command[command.index('--output')+1])
                if run not in (root/'raw'/'seed1', root/'gsd'/'seed1'):
                    continue
                if (run/'warm/complete.json').exists() or (run/'warm/stopped_early.json').exists():
                    continue
                try:
                    valid = checked_process(int(proc.name),run,allocation)
                except RuntimeError as error:
                    if 'worker' in str(error):
                        continue
                    raise
                if not valid:
                    continue
                print(f'Adopting existing trainer PID{proc.name}: {run}', flush=True)
                monitor(int(proc.name), run, allocation, policy)
                found = True
                break
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not found:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--allocation', required=True)
    parser.add_argument('--node', default='hpc23')
    parser.add_argument('--adopt-pair', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') != args.allocation or subprocess.check_output(['hostname','-s'], text=True).strip() != args.node:
        raise RuntimeError('This controller must run in the authorized interactive on '+args.node)
    policy = json.loads(args.policy.read_text())
    scout = policy.get('mode') == 'no_improvement'
    if (policy['patience_evaluations'] < 2 or policy['min_updates'] < policy['eval_period'] or
            (not scout and (not 0 < policy['below_peak_margin'] < 1 or not 0 < policy['consecutive_decline_margin'] < 1)) or
            (scout and (policy['patience_evaluations'] != 2 or policy['eval_period'] != 1000)) or
            policy['max_updates'] <= policy['min_updates']):
        raise ValueError('Invalid early-stopping policy')
    if not isinstance(policy.get('require_continuing_decline', True), bool):
        raise ValueError('require_continuing_decline must be a boolean')
    if args.adopt_pair:
        adopt_pair(args.adopt_pair, args.state, args.allocation, policy)
        return
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not args.run or not command:
        parser.error('Managed training requires --run and a command')
    try:
        from scripts.dtst_deferred_run import deferred_run
    except ModuleNotFoundError:
        from dtst_deferred_run import deferred_run
    if deferred_run(args.run):
        print(f'Deferred unstarted run {args.run}; prioritize method pilots and conditional controls.',flush=True)
        return
    if (args.run/'priority_before_training.json').exists():
        try:
            from scripts.dtst_method_priority import priority_callback
        except ModuleNotFoundError:
            from dtst_method_priority import priority_callback
        priority_callback(args.run,'before')
        if deferred_run(args.run):return
    process = subprocess.Popen(command)
    monitor(process.pid, args.run, args.allocation, policy)
    code = process.wait()
    if code and not (args.run/'warm/stopped_early.json').exists():
        raise SystemExit(code if code > 0 else 1)
    if (args.run/'priority_after_training.json').exists():
        try:
            from scripts.dtst_method_priority import priority_callback
        except ModuleNotFoundError:
            from dtst_method_priority import priority_callback
        priority_callback(args.run,'after')


if __name__ == '__main__':
    main()
