"""Scientific stop parity and checkpoint-safe session pauses, without a GPU."""
import json,os,subprocess,sys,tempfile,time,unittest,uuid
from pathlib import Path
from scripts.run_dtst_kaggle import scientific_stop,trim_uncommitted
from scripts.dtst_run_control import checked_process,stop_at_checkpoint

class KagglePolicyTests(unittest.TestCase):
    def test_rising_cap_ignores_old_patience_until_first_decline(self):
        root=Path(__file__).parents[1]
        policy=json.loads((root/'configs/dtst_sfuda/early_stopping_scout_2eval.json').read_text())
        budget=dict(name='dtst_recovery_budget_20261016_v1',phase='dino_recovery_open',max_updates=9000,
            original_policy=policy,user_requested=True,applied_after_training_started=False,
            rising_cap_rule='new_peak_at_cap_then_first_strict_decline_v1')
        curve=[dict(iteration=i,metrics={'miou':v}) for i,v in [(0,.52),(6000,.58),(7000,.579),(8000,.578),(9000,.581),(10000,.581)]]
        self.assertIsNone(scientific_stop(curve,policy,budget))
        curve.append(dict(iteration=11000,metrics={'miou':.5809}))
        self.assertEqual(scientific_stop(curve,policy,budget)['iteration'],11000)
    def test_pause_trims_uncommitted_updates_but_keeps_backup(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            run=Path(directory);(run/'warm').mkdir()
            path=run/'warm/training_history.jsonl';path.write_text('{"iteration":1000}\n{"iteration":1001}\n')
            trim_uncommitted(run,1000)
            self.assertEqual(json.loads(path.read_text())['iteration'],1000)
            backup=next((run/'session_history_archive').glob('*'))
            self.assertIn('1001',backup.read_text())

@unittest.skipUnless(Path('/proc').exists(),'Linux process checks')
class KaggleCheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;self.temp=tempfile.TemporaryDirectory(dir=Path.cwd());self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'run';self.stage=self.run/'warm';self.stage.mkdir(parents=True)
        self.token=uuid.uuid4().hex;self.identity='local:'+self.token
        script=self.root/'train_DTST_remote.py';script.write_text('import time\ntime.sleep(120)\n')
        self.child=subprocess.Popen([sys.executable,str(script),'--output',str(self.run)],
            env=dict(os.environ,DTST_LOCAL_RUN_ID=self.token))
        self.addCleanup(self.cleanup);time.sleep(.05)
    def cleanup(self):
        if self.child.poll() is None:self.child.terminate()
        self.child.wait(timeout=10)
    def test_identity_and_output_guard(self):
        self.assertTrue(checked_process(self.child.pid,self.run,self.identity))
        with self.assertRaises(RuntimeError):checked_process(self.child.pid,self.run,'local:wrong')
        with self.assertRaises(RuntimeError):checked_process(self.child.pid,self.root/'other',self.identity)
    def test_time_pause_keeps_full_state_and_is_not_numerical_stop(self):
        metadata=dict(iteration=1000,protocol_signature='protocol',source_checkpoint_sha256='source')
        for name in ('last.pth','resume.pth','best_target_val.pth'):self.torch.save(metadata,self.stage/name)
        (self.run/'protocol.json').write_text(json.dumps(metadata))
        (self.stage/'training_history.jsonl').write_text('{"iteration":1000}\n')
        decision=dict(iteration=1000,peak_iteration=1000,peak_miou=.6,reason='session time budget')
        self.assertTrue(stop_at_checkpoint(self.child.pid,self.run,self.identity,decision,{'max_updates':20000},stop_kind='session_pause'))
        marker=json.loads((self.stage/'session_pause.json').read_text())
        self.assertFalse(marker['target_labels_used_for_stopping'])
        self.assertFalse((self.stage/'stopped_early.json').exists())
        self.assertEqual(self.torch.load(self.stage/'resume.pth',weights_only=False)['iteration'],1000)
    def test_full_endpoint_is_never_relabelled_as_a_session_pause(self):
        decision=dict(iteration=20000,peak_iteration=20000)
        self.assertFalse(stop_at_checkpoint(self.child.pid,self.run,self.identity,decision,{'max_updates':20000},stop_kind='session_pause'))
        self.assertIsNone(self.child.poll())
        (self.stage/'complete.json').write_text('{}')
        decision['iteration']=1000
        self.assertFalse(stop_at_checkpoint(self.child.pid,self.run,self.identity,decision,{'max_updates':20000},stop_kind='session_pause'))

if __name__=='__main__':unittest.main()
