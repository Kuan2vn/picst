"""Execute only the registered same-allocation priority callback."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

RUNNER=Path('/tmp/gsd_dtst_84806_20010752/method_priority_20261015/run_priority_hpc.py')

def priority_callback(run,when):
    path=Path(run)/f'priority_{when}_training.json'
    if not path.exists():return
    declaration=json.loads(path.read_text())
    expected={'before':'gsd_blend','after':'gsd_blend_lr1e5'}
    if (declaration.get('name')!='main_method_priority_20261015_v1' or
            Path(run).parent.name!=expected[when] or declaration.get('output')!=str(run) or
            declaration.get('runner')!=str(RUNNER) or
            hashlib.sha256(RUNNER.read_bytes()).hexdigest()!=declaration.get('runner_sha256')):
        raise ValueError('Unregistered method-priority callback')
    subprocess.run([sys.executable,str(RUNNER),'--when',when,'--trigger-run',str(run)],check=True)
