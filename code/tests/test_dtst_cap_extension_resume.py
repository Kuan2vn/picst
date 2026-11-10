import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipIf(os.name=='nt','Native Linux controller uses fcntl')
class CapResumeTests(unittest.TestCase):
    def test_reopen_preserves_finalized_stop_and_full_state(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'artifacts/dtst_scout_20261015/run_rising_cap_extensions.py'
        if not path.exists():path=Path('/tmp/gsd_dtst_84806_20010752/scout_20261015/run_rising_cap_extensions.py')
        spec=importlib.util.spec_from_file_location('cap_resume_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        policy=json.loads((root/'configs/dtst_sfuda/early_stopping_scout_2eval.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);warm=run/'warm';warm.mkdir()
            def write(path,data):path.write_text(json.dumps(data))
            budget=dict(name='dtst_scout_budget_20261015_v1',phase='support_hard',max_updates=9000,
                        original_policy=policy,user_requested=True,applied_after_training_started=False)
            write(run/'pilot_budget.json',budget);write(run/'early_stopping_policy.json',policy)
            write(run/'protocol.json',dict(protocol_signature='test'))
            stop=dict(iteration=9000,protocol_signature='test',pilot_budget_cap=True)
            for file,key in [('last.pth','checkpoint_sha256'),('resume.pth','resume_checkpoint_sha256'),
                             ('best_target_val.pth','peak_checkpoint_sha256')]:
                (warm/file).write_bytes(file.encode());stop[key]=module.digest(warm/file)
            write(warm/'stopped_early.json',stop)
            write(warm/'early_stop_intent.json',stop)
            (warm/'evaluation_curve.jsonl').write_text('\n'.join(json.dumps(dict(iteration=i,metrics={'miou':v}))
                                                      for i,v in [(0,.5),(8000,.6),(9000,.61)]))
            with patch.object(module,'refresh_budget_guards') as refresh:
                self.assertTrue(module.reopen(run));refresh.assert_called_once()
            self.assertFalse((warm/'stopped_early.json').exists())
            self.assertEqual(json.loads((warm/'before_rising_cap_extension/stopped_early.json').read_text()),stop)
            for file,key in [('last.pth','checkpoint_sha256'),('resume.pth','resume_checkpoint_sha256')]:
                self.assertEqual(module.digest(warm/file),stop[key])
            self.assertTrue(module.reopen(run))  # interrupted continuation is resumable


if __name__=='__main__':unittest.main()
