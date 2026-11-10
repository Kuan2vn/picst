"""Explicitly defer one unstarted baseline; never invent completion metrics."""
import json
from pathlib import Path

SOURCE_SHA='10fbe026b5f2dcef331a99d97bab551f7cb0a2cf95feac2625b82ca3e3449d31'

def deferred_run(run):
    run=Path(run)
    path=run/'defer_training.json'
    if not path.exists():
        return None
    declaration=json.loads(path.read_text())
    caps={'raw_lr1e5':12000,'gsd_blend':9000,'gsd_blend_lr1e5':12000}
    phase=declaration.get('phase')
    if (declaration.get('name')!='main_method_priority_20261015_v1' or phase not in caps or
            run.name!='seed1' or run.parent.name!=phase or
            declaration.get('output')!=str(run) or declaration.get('user_requested_priority') is not True or
            declaration.get('source_checkpoint_sha256')!=SOURCE_SHA):
        raise ValueError('Invalid baseline deferral declaration')
    budget=json.loads((run/'pilot_budget.json').read_text())
    if budget['phase']!=phase or budget['max_updates']!=caps[phase]:
        raise ValueError('Deferred baseline differs from its declared pilot')
    release=run/'release_deferred_run.json'
    if release.exists():
        value=json.loads(release.read_text())
        if (phase!='raw_lr1e5' or value.get('name')!=declaration['name'] or
                value.get('screen_passed') is not True or value.get('screen_gain_pp',0)<.5 or
                value.get('source_checkpoint_sha256')!=SOURCE_SHA):
            raise ValueError('Invalid conditional baseline release')
        return None
    if (run/'protocol.json').exists() or any((run/'warm').glob('*.pth')):
        raise ValueError('Cannot defer a baseline that has already started')
    return declaration
