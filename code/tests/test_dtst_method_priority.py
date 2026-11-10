import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.dtst_method_priority import RUNNER, priority_callback
from scripts.dtst_pilot_budget import validate_budget
from scripts.dtst_deferred_run import deferred_run
from tests import test_dtst_deferred_run as deferred_fixture

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('priority_runner',ROOT/'artifacts/dtst_method_priority_20261015/run_priority_hpc.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class PriorityTests(unittest.TestCase):
    def run_before(self,candidate_miou,matched_miou=None):
        point=lambda m:dict(iteration=9000,metrics={'miou':m,'iou_by_class':{'car':.8}})
        raw_curve=[point(.6)]
        responses=[(({},[point(candidate_miou)]),runner.OUTPUT/'gsd_hard_readout1/seed1')]
        if matched_miou is not None:responses.append((({},[point(matched_miou)]),runner.OUTPUT/'raw_readout1/seed1'))
        trigger=runner.FOLLOW/'gsd_blend/seed1'
        with patch('sys.argv',['priority','--when','before','--trigger-run',str(trigger)]),\
             patch.dict('os.environ',{'SLURM_JOB_ID':'86126'}),\
             patch.object(runner.subprocess,'check_output',return_value='hpc23\n'),\
             patch.object(runner,'check_registry'),patch.object(runner,'summarize',return_value=({},raw_curve)),\
             patch.object(runner,'run_one',side_effect=responses) as execute,\
             patch.object(runner,'state') as state,patch.object(runner,'defer') as defer,\
             patch.object(runner.subprocess,'run'):
            runner.main()
            return execute.call_args_list,state.call_args_list,defer.call_count

    def test_failed_screen_never_spends_gpu_on_a_matched_raw(self):
        calls,states,deferred=self.run_before(.604)
        self.assertEqual(len(calls),1)
        self.assertEqual(states[-1].args[1],'diagnostic_fallback')
        self.assertEqual(deferred,0)

    def test_screen_and_matched_success_defer_diagnostics(self):
        calls,states,deferred=self.run_before(.607,.603)
        self.assertEqual([c.args[0] for c in calls],['gsd_hard_readout1','raw_readout1'])
        self.assertTrue(states[-1].kwargs['confirmed'])
        self.assertEqual(deferred,2)

    def test_recipe_only_gain_returns_to_diagnostics(self):
        calls,states,deferred=self.run_before(.607,.606)
        self.assertEqual(len(calls),2)
        self.assertEqual(states[-1].args[1],'diagnostic_fallback')
        self.assertEqual(deferred,0)

    def test_gate_rejects_small_gain_and_car_collapse(self):
        point=lambda m,c:dict(metrics={'miou':m,'iou_by_class':{'car':c}})
        ref=point(.6,.8)
        self.assertTrue(runner.qualifies(ref,point(.605,.78),.5))
        self.assertFalse(runner.qualifies(ref,point(.6049,.8),.5))
        self.assertFalse(runner.qualifies(ref,point(.61,.779),.5))

    def test_priority_caps_require_readout_multiplier1(self):
        policy=json.loads((ROOT/'configs/dtst_sfuda/early_stopping_1k.json').read_text())
        recipe=json.loads((ROOT/'artifacts/dtst_validation1k_20261014/recipe_b2_irrg_eval1k.json').read_text())
        for phase,arm in [('gsd_hard_readout1','gsd'),('raw_readout1','raw')]:
            budget=dict(name='dtst_pilot_caps_20261015_v1',phase=phase,max_updates=9000,original_policy=policy,
                        user_requested=True,applied_after_training_started=False)
            protocol=dict(recipe=recipe,gsd_study={'arm':arm})
            with self.assertRaisesRegex(ValueError,'readout'):validate_budget(budget,policy,protocol)
            protocol['recipe']=dict(recipe,readout_lr_mult=1)
            self.assertEqual(validate_budget(budget,policy,protocol),9000)

    def test_callback_only_executes_the_registered_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'gsd_blend/seed1';run.mkdir(parents=True)
            marker=run/'priority_before_training.json'
            declaration=dict(name='main_method_priority_20261015_v1',output=str(run),runner=str(RUNNER),
                             runner_sha256=hashlib.sha256(b'frozen').hexdigest())
            marker.write_text(json.dumps(declaration))
            with patch('pathlib.Path.read_bytes',return_value=b'frozen'),patch('subprocess.run') as execute:
                priority_callback(run,'before');self.assertEqual(execute.call_count,1)
            marker.write_text(json.dumps(dict(declaration,runner='/another/script.py')))
            with patch('subprocess.run') as execute:
                with self.assertRaises(ValueError):priority_callback(run,'before')
                execute.assert_not_called()

    def test_conditional_release_restores_a_deferred_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            run=deferred_fixture.DeferredRunTests().fixture(directory)
            declaration=json.loads((run/'defer_training.json').read_text())
            release=dict(name=declaration['name'],source_checkpoint_sha256=declaration['source_checkpoint_sha256'],
                         screen_passed=True,screen_gain_pp=.6)
            (run/'release_deferred_run.json').write_text(json.dumps(release))
            self.assertIsNone(deferred_run(run))
            (run/'release_deferred_run.json').write_text(json.dumps(dict(release,screen_gain_pp=.1)))
            with self.assertRaises(ValueError):deferred_run(run)

if __name__=='__main__':unittest.main()
