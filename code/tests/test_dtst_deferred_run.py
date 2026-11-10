import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.dtst_deferred_run import SOURCE_SHA, deferred_run
from scripts.dtst_run_control import main as control_main
from scripts.summarize_dtst_results import main as report_main

class DeferredRunTests(unittest.TestCase):
    def fixture(self,root):
        run=Path(root)/'raw_lr1e5/seed1'
        run.mkdir(parents=True)
        (run/'pilot_budget.json').write_text(json.dumps(dict(phase='raw_lr1e5',max_updates=12000)))
        declaration=dict(name='main_method_priority_20261015_v1',phase='raw_lr1e5',output=str(run),
                         user_requested_priority=True,source_checkpoint_sha256=SOURCE_SHA)
        (run/'defer_training.json').write_text(json.dumps(declaration))
        return run

    def test_deferral_never_launches_a_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.fixture(directory)
            policy=Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping_1k.json'
            args=['control','--policy',str(policy),'--allocation','86126','--run',str(run),'--','not-a-real-trainer']
            with patch('sys.argv',args),patch.dict('os.environ',{'SLURM_JOB_ID':'86126'}),\
                 patch('subprocess.check_output',return_value='hpc23\n'),patch('subprocess.Popen') as start:
                control_main()
                start.assert_not_called()

    def test_reporting_deferral_is_not_completion_or_a_peak(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.fixture(directory)
            output=Path(directory)/'report'
            with patch('sys.argv',['report',str(run),'--output',str(output)]):report_main()
            result=json.loads((output/'DEFERRED.json').read_text())
            self.assertFalse(result['completed'])
            self.assertIsNone(result['peak'])
            self.assertFalse((output/'peak_and_endpoint.json').exists())

    def test_started_or_wrong_baseline_cannot_be_deferred(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.fixture(directory)
            (run/'protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'already started'):deferred_run(run)
            (run/'protocol.json').unlink()
            data=json.loads((run/'defer_training.json').read_text())
            data['phase']='raw'
            (run/'defer_training.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'declaration'):deferred_run(run)

if __name__=='__main__':unittest.main()
