import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.dtst_pilot_budget import cap_decision, cap_extension_active, effective_policy, validate_budget, RISING_CAP_RULE
from scripts.summarize_dtst_results import summarize


class PilotBudgetTests(unittest.TestCase):
    policy=json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping_1k.json').read_text())

    def budget(self):
        return dict(name='dtst_pilot_caps_20261015_v1',phase='confidence_blend',max_updates=9000,
                    original_policy=self.policy,user_requested=True,applied_after_training_started=True)

    def points(self,stop=9000,declining=False):
        return [dict(iteration=i,protocol_signature='matched',model='EMA',image_count=2016,
                     metrics={'miou': .6 if i==3000 else .59 if declining and i>=5000 else
                              .599+i/100000000 if i>3000 else .5+i/1000000})
                for i in range(0,stop+1,1000)]

    def test_exact_saved_cap_and_peak(self):
        points=self.points()
        self.assertIsNone(cap_decision(points[:-1],self.budget(),self.policy))
        decision=cap_decision(points,self.budget(),self.policy)
        self.assertEqual(decision['iteration'],9000)
        self.assertEqual(decision['peak_iteration'],3000)
        with self.assertRaisesRegex(RuntimeError,'missed'):
            cap_decision(points+[dict(iteration=10000,metrics={'miou':.6})],self.budget(),self.policy)
        with self.assertRaises(ValueError):
            validate_budget(dict(self.budget(),max_updates=6000),self.policy)

    def test_rising_cap_continues_ties_and_stops_first_decline(self):
        budget=dict(self.budget(),rising_cap_rule=RISING_CAP_RULE)
        curve=self.points()
        curve[-1]['metrics']['miou']=.61
        self.assertTrue(cap_extension_active(curve,budget,self.policy))
        self.assertIsNone(cap_decision(curve,budget,self.policy))
        for step,value in [(10000,.62),(11000,.62),(12000,.62)]:
            curve.append(dict(iteration=step,metrics={'miou':value}))
            self.assertIsNone(cap_decision(curve,budget,self.policy))
        curve.append(dict(iteration=13000,metrics={'miou':.61999}))
        stop=cap_decision(curve,budget,self.policy)
        self.assertEqual(stop['iteration'],13000)
        self.assertTrue(stop['pilot_cap_extension'])
        self.assertEqual(stop['peak_iteration'],10000)

    def test_tie_at_cap_does_not_qualify_and_safety_ceiling(self):
        budget=dict(self.budget(),rising_cap_rule=RISING_CAP_RULE)
        curve=self.points();curve[-1]['metrics']['miou']=.6
        self.assertFalse(cap_extension_active(curve,budget,self.policy))
        self.assertTrue(cap_decision(curve,budget,self.policy)['pilot_budget_cap'])
        curve[-1]['metrics']['miou']=.61
        curve.append(dict(iteration=20000,metrics={'miou':.60}))
        self.assertIsNone(cap_decision(curve,budget,self.policy))

    def test_report_accepts_first_decline_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'extended.zip'
            budget=dict(self.budget(),rising_cap_rule=RISING_CAP_RULE)
            self.report_fixture(path,budget=budget,marker_budget=budget)
            with zipfile.ZipFile(path) as z:files={n:z.read(n) for n in z.namelist()}
            curve=[json.loads(line) for line in files['warm/evaluation_curve.jsonl'].splitlines()]
            curve[-1]['metrics']['miou']=.61
            for step,value in [(10000,.62),(11000,.619)]:
                point=dict(curve[-1],iteration=step,metrics={'miou':value});curve.append(point)
            marker=json.loads(files['warm/stopped_early.json'])
            marker.pop('pilot_budget_cap');marker.update(cap_decision(curve,budget,self.policy))
            files['warm/stopped_early.json']=json.dumps(marker).encode()
            files['warm/observed_peak.json']=json.dumps(curve[-2]).encode()
            files['warm/evaluation_curve.jsonl']='\n'.join(map(json.dumps,curve)).encode()
            with zipfile.ZipFile(path,'w') as z:
                for name,value in files.items():z.writestr(name,value)
            report,_=summarize(path)
            self.assertEqual(report['stopping'],'pilot_extension_first_decline')
            self.assertEqual(report['actual_updates'],11000)
            self.assertEqual(report['peak']['iteration'],10000)

    def report_fixture(self,path,cap_stop=True,budget=None,marker_budget=None):
        budget=budget or self.budget()
        points=self.points(9000 if cap_stop else 7000,declining=not cap_stop)
        protocol=dict(protocol_signature='matched',seed=1,source_checkpoint_sha256='source',manifests={},
                      gsd_study={'arm':'confidence'},recipe={'batch_size':6,'base_lr':2e-5,
                      'eval_period':1000,'warm':{'steps':20000,'lr_max_iter':32000}})
        marker=dict(iteration=points[-1]['iteration'],planned_iteration=20000,
                    protocol_signature='matched',stopping_policy=self.policy)
        if cap_stop:
            marker.update(cap_decision(points,self.budget(),self.policy))
            if marker_budget is not None:marker['pilot_budget']=marker_budget
        with zipfile.ZipFile(path,'w') as z:
            for name,value in [('protocol.json',protocol),('early_stopping_policy.json',self.policy),
                ('pilot_budget.json',budget),('warm/stopped_early.json',marker),('warm/observed_peak.json',points[3])]:
                z.writestr(name,json.dumps(value))
            z.writestr('warm/evaluation_curve.jsonl','\n'.join(map(json.dumps,points)))

    def test_report_accepts_cap_or_earlier_numerical_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            for capped in (True,False):
                path=Path(directory)/f'{capped}.zip'
                self.report_fixture(path,cap_stop=capped)
                report,_=summarize(path)
                self.assertEqual(report['stopping'],'pilot_budget_cap' if capped else 'target_validation_early_stopped')
                self.assertEqual(report['stopping_policy'],effective_policy(self.policy,self.budget()))
                self.assertEqual(report['peak']['iteration'],3000)
                self.assertFalse(report['endpoint']['is_fixed_budget_endpoint'])

    def test_report_rejects_cap_marker_or_recipe_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            for name,budget,marker in [('phase',dict(self.budget(),phase='gsd_blend'),None),
                    ('marker',self.budget(),dict(self.budget(),max_updates=12000))]:
                path=Path(directory)/(name+'.zip')
                self.report_fixture(path,budget=budget,marker_budget=marker)
                with self.assertRaises(ValueError):summarize(path)


if __name__=='__main__':unittest.main()
