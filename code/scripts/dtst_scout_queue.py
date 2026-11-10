"""One registered continuation after the current experiment, in its existing allocation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

RUNNER=Path('/tmp/gsd_dtst_84806_20010752/scout_20261015/run_scout_hpc.py')

def after_training(run):
    geometry = Path(run)/'geometry_after_training.json'
    if geometry.exists():
        record = json.loads(geometry.read_text())
        runner = Path('/tmp/gsd_dtst_84806_20010752/geometry_20261016/run_geometry_hpc.py')
        expected = Path('/tmp/gsd_dtst_84806_20010752/local_results/dtst_study/scout_20261015/generic_support_bank/seed1')
        if (Path(run) != expected or record.get('name') != 'dtst_geometry_priority_20261016_v1' or
                record.get('output') != str(run) or record.get('runner') != str(runner) or
                hashlib.sha256(runner.read_bytes()).hexdigest() != record.get('runner_sha256')):
            raise ValueError('Unregistered geometry continuation')
        subprocess.run([sys.executable, '-u', str(runner), '--trigger-run', str(run)], check=True)
    extension=Path(run)/'rising_cap_continuation.json'
    if extension.exists():
        record=json.loads(extension.read_text())
        runner=RUNNER.parent/'run_rising_cap_extensions.py'
        expected=Path('/tmp/gsd_dtst_84806_20010752/local_results/dtst_study/scout_20261015/raw_readout1_scout/seed1')
        if (Path(run)!=expected or record.get('output')!=str(run) or
                record.get('name')!='rising_cap_continuation_20261015_v1' or
                record.get('runner')!=str(runner) or
                hashlib.sha256(runner.read_bytes()).hexdigest()!=record.get('runner_sha256')):
            raise ValueError('Unregistered rising-cap continuation')
        subprocess.run([sys.executable,'-u',str(runner),'--trigger-run',str(run)],check=True)
    path=Path(run)/'scout_after_training.json'
    if not path.exists():return
    record=json.loads(path.read_text())
    expected=Path('/tmp/gsd_dtst_84806_20010752/local_results/dtst_study/method_priority_20261015/gsd_hard_readout1/seed1')
    if (Path(run)!=expected or record.get('name')!='gsd_scout_priority_20261015_v1' or
            record.get('runner')!=str(RUNNER) or record.get('output')!=str(run) or
            hashlib.sha256(RUNNER.read_bytes()).hexdigest()!=record.get('runner_sha256')):
        raise ValueError('Unregistered scout continuation')
    subprocess.run([sys.executable,'-u',str(RUNNER),'--trigger-run',str(run)],check=True)
