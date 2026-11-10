"""Compact progress and timing for the isolated source-initialized DT-ST study."""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import tempfile


def rows(path):
    if not path.is_file(): return []
    text=path.read_text(); lines=text.splitlines(); values=[]
    for index,line in enumerate(lines):
        try: values.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1 or text.endswith('\n'): raise
    return values


def snapshot(root):
    study = root/'results/dtst_study'
    runs = []
    for protocol_path in sorted(study.glob('**/protocol.json')):
        protocol = json.loads(protocol_path.read_text())
        path = protocol_path.parent/'warm'
        history = rows(path/'training_history.jsonl')
        curve = rows(path/'evaluation_curve.jsonl')
        complete = json.loads((path/'complete.json').read_text()) if (path/'complete.json').is_file() else None
        last = history[-1] if history else {}
        runs.append({'run':str(protocol_path.parent.relative_to(study)), 'complete':bool(complete),
                     'iteration':complete['iteration'] if complete else last.get('iteration',0),
                     'seconds':complete.get('wall_seconds') if complete else None,
                     'last':last, 'curve':curve, 'source':protocol['source_checkpoint_sha256'],
                     'settings':protocol['dtst'], 'selector':protocol['gsd_study'],
                     'scope':protocol['gsd_scope']})
    result={'updated_bangkok':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(), 'runs':runs}
    study.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=study,prefix='.snapshot-',delete=False) as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    Path(stream.name).replace(study/'snapshot.json')
    print(json.dumps({'updated':result['updated_bangkok'],'runs':len(runs)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--watch-pid',type=int)
    args=parser.parse_args()
    while True:
        snapshot(args.root)
        if not args.watch_pid: break
        try: os.kill(args.watch_pid,0)
        except ProcessLookupError: break
        time.sleep(600)
