import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.summarize_dtst_results import summarize


class ReportingTests(unittest.TestCase):
    def fixture(self, path, mismatch=False, completed=6000):
        points = [dict(iteration=step, protocol_signature='matched', model='EMA', image_count=2016,
                       metrics=dict(miou=miou, iou_by_class={'car': car, 'tree': tree}))
                  for step, miou, car, tree in [(0, .4, .8, .2), (3000, .65, .3, .9), (6000, .65, .8, .5)]]
        if mismatch:
            points[-1]['protocol_signature'] = 'different'
        protocol = dict(protocol_signature='matched', seed=1, recipe={'warm': {'steps': 6000}},
                        source_checkpoint_sha256='source', manifests={}, gsd_study={'arm': 'gsd'})
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('protocol.json', json.dumps(protocol))
            archive.writestr('warm/complete.json', json.dumps(dict(iteration=completed, protocol_signature='matched')))
            archive.writestr('warm/observed_peak.json', json.dumps(points[1]))
            archive.writestr('warm/evaluation_curve.jsonl', '\n'.join(map(json.dumps, points)))

    def test_peak_is_one_checkpoint_and_ties_choose_earliest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.zip'
            self.fixture(path)
            report, _ = summarize(path)
            self.assertEqual(report['peak']['iteration'], 3000)
            self.assertEqual(report['peak']['metrics']['iou_by_class']['car'], .3)
            self.assertEqual(report['endpoint']['metrics']['iou_by_class']['car'], .8)

    def test_rejects_mixed_protocol_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.zip'
            self.fixture(path, mismatch=True)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                summarize(path)

    def test_rejects_incomplete_fixed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.zip'
            self.fixture(path, completed=3000)
            with self.assertRaisesRegex(ValueError, 'fixed adaptation budget'):
                summarize(path)

    def test_declared_early_stop_reports_actual_budget(self):
        policy = json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping.json').read_text())
        points = [dict(iteration=step, protocol_signature='matched', model='EMA', image_count=2016,
                       metrics=dict(miou=value)) for step,value in [(0,.5),(3000,.55),(6000,.6),(9000,.595),(12000,.58)]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'early.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('protocol.json',json.dumps(dict(protocol_signature='matched',seed=1,
                    recipe={'warm':{'steps':20000}},source_checkpoint_sha256='source',manifests={},gsd_study={'arm':'gsd'})))
                archive.writestr('early_stopping_policy.json',json.dumps(policy))
                archive.writestr('warm/stopped_early.json',json.dumps(dict(iteration=12000,planned_iteration=20000,
                    protocol_signature='matched',stopping_policy=policy)))
                archive.writestr('warm/observed_peak.json',json.dumps(points[2]))
                archive.writestr('warm/evaluation_curve.jsonl','\n'.join(map(json.dumps,points)))
            report,_ = summarize(path)
            self.assertEqual(report['actual_updates'],12000)
            self.assertEqual(report['peak']['iteration'],6000)
            self.assertTrue(report['target_labels_used_for_stopping'])
            self.assertFalse(report['endpoint']['is_fixed_budget_endpoint'])

    def test_1k_policy_reports_a_low_plateau_stop(self):
        policy = json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping_1k.json').read_text())
        points = [dict(iteration=step,protocol_signature='matched',model='EMA',image_count=2016,
                       metrics={'miou':value}) for step,value in
                  [(0,.52),(1000,.56),(2000,.57),(3000,.6),(4000,.599),(5000,.595),(6000,.5955),(7000,.596)]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'early_1k.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('protocol.json',json.dumps(dict(protocol_signature='matched',seed=1,
                    recipe={'warm':{'steps':20000},'eval_period':1000},source_checkpoint_sha256='source',
                    manifests={},gsd_study={'arm':'gsd'})))
                archive.writestr('early_stopping_policy.json',json.dumps(policy))
                archive.writestr('warm/stopped_early.json',json.dumps(dict(iteration=7000,planned_iteration=20000,
                    protocol_signature='matched',stopping_policy=policy)))
                archive.writestr('warm/observed_peak.json',json.dumps(points[3]))
                archive.writestr('warm/evaluation_curve.jsonl','\n'.join(map(json.dumps,points)))
            report,_ = summarize(path)
            self.assertEqual(report['actual_updates'],7000)
            self.assertEqual(report['peak']['iteration'],3000)
            self.assertEqual(report['evaluation_updates'],list(range(0,7001,1000)))
            self.assertFalse(report['endpoint']['is_fixed_budget_endpoint'])

    def test_user_requested_cap_is_explicit_and_does_not_bypass_invalid_metadata(self):
        policy=json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping.json').read_text())
        points=[dict(iteration=i,protocol_signature='matched',model='EMA',image_count=2016,
                     metrics={'miou':m}) for i,m in [(0,.5),(3000,.55),(6000,.6),(9000,.598),(12000,.595)]]
        override=dict(requested_stop_update=12000,original_policy=policy,posthoc_budget_override=True)
        effective=dict(policy,user_requested_budget_cap=12000,posthoc_budget_override=True)
        with tempfile.TemporaryDirectory() as directory:
            for name,metadata in [('valid',override),('wrong_cap',dict(override,requested_stop_update=9000)),
                                  ('missing',None)]:
                path=Path(directory)/(name+'.zip')
                with zipfile.ZipFile(path,'w') as z:
                    z.writestr('protocol.json',json.dumps(dict(protocol_signature='matched',seed=1,
                        recipe={'warm':{'steps':20000}},source_checkpoint_sha256='source',manifests={},
                        gsd_study={'arm':'generic'})))
                    z.writestr('early_stopping_policy.json',json.dumps(policy))
                    z.writestr('warm/stopped_early.json',json.dumps(dict(iteration=12000,planned_iteration=20000,
                        protocol_signature='matched',stopping_policy=effective,user_requested_stop=True)))
                    z.writestr('warm/observed_peak.json',json.dumps(points[2]))
                    z.writestr('warm/evaluation_curve.jsonl','\n'.join(map(json.dumps,points)))
                    if metadata is not None:
                        z.writestr('warm/user_stop_override.json',json.dumps(metadata))
                if name!='valid':
                    with self.assertRaisesRegex(ValueError,'override'):
                        summarize(path)
                else:
                    report,_=summarize(path)
                    self.assertEqual(report['stopping'],'user_requested_budget_cap')
                    self.assertEqual(report['peak']['iteration'],6000)
                    self.assertFalse(report['endpoint']['is_fixed_budget_endpoint'])
                    self.assertNotEqual(report['stopping_policy'],policy)


if __name__ == '__main__':
    unittest.main()
