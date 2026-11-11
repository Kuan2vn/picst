#!/usr/bin/env python3
"""Collect every run's result files into one small folder (results/) plus results.zip, to send to the supervisor.

  python collect_results.py            # reads ./runs, writes ./results/ and ./results.zip

results/summary.md lists each run with its status and peak / last mIoU6 (also for runs still in progress).
Per run it copies report/PEAK_AND_ENDPOINT.md, report/peak_and_endpoint.json and RUN_STATE.json (no checkpoints).
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path


def curve_points(run):
    for stage in ('warm', 'snd'):
        f = run/stage/'evaluation_curve.jsonl'
        if f.is_file():
            pts = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            return [(p['iteration'], 100*p['metrics']['miou']) for p in pts]
    return []


def summarize(run):
    state = {}
    if (run/'RUN_STATE.json').is_file():
        state = json.loads((run/'RUN_STATE.json').read_text())
    pts = curve_points(run)
    row = dict(status=state.get('status', 'unknown' if pts else 'not started'), peak='', last='', evals=max(len(pts)-1, 0))
    rep = run/'report/peak_and_endpoint.json'
    if rep.is_file():  # final numbers
        j = json.loads(rep.read_text())
        if 'peak' in j:  # baseline runner format
            peak, last = (j['peak']['iteration'], j['peak']['miou6']), (j['last']['iteration'], j['last']['miou6'])
        else:  # DT-ST / PIC-ST format: {arm_seed: report}
            j = next(iter(j.values()))
            peak = (j['peak']['iteration'], 100*j['peak']['metrics']['miou'])
            last = (j['endpoint']['iteration'], 100*j['endpoint']['metrics']['miou'])
    elif pts:  # in progress: from the evaluation curve so far
        peak, last = max(pts, key=lambda p: p[1]), pts[-1]
    else:
        return row
    row.update(peak=f'{peak[1]:.2f} ({peak[0]})', last=f'{last[1]:.2f} ({last[0]})')
    return row


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=here/'runs')
    ap.add_argument('--out', type=Path, default=here/'results')
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    a.out.mkdir(parents=True)
    lines = ['| setting | experiment | status | evals done | peak mIoU6 (iter) | last mIoU6 (iter) |',
             '|---|---|---|---:|---|---|']
    for run in sorted(a.runs.glob('*/*/seed1')):
        setting, exp = run.parent.parent.name, run.parent.name
        r = summarize(run)
        lines.append(f"| {setting} | {exp} | {r['status']} | {r['evals']} | {r['peak']} | {r['last']} |")
        dest = a.out/setting/exp
        for rel in ('report/PEAK_AND_ENDPOINT.md', 'report/peak_and_endpoint.json', 'RUN_STATE.json'):
            if (run/rel).is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run/rel, dest/Path(rel).name)
    note = ('\nPeak is selected with target-eval labels (disclosed); "last" is the latest evaluation. '
            'Unfinished runs show the curve so far.\n')
    (a.out/'summary.md').write_text('\n'.join(lines)+'\n'+note, encoding='utf-8', newline='\n')
    with zipfile.ZipFile(here/'results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.out.rglob('*')):
            if f.is_file():
                z.write(f, f.relative_to(a.out.parent))
    print('\n'.join(lines))
    print(f'\nWrote {a.out} and {here/"results.zip"}')


if __name__ == '__main__':
    main()
