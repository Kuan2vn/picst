"""CPU checks for the baseline runner: budget, stopping parity with the PIC-ST arms, warm verification, command."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from scripts import run_baseline_kaggle as rb
from scripts.dtst_pilot_budget import RISING_CAP_RULE

ROOT=Path(__file__).parents[1]


def policy():
    for path in (ROOT/'kaggle_settings/early_stopping_scout_2eval.json',ROOT/'configs/dtst_sfuda/early_stopping_scout_2eval.json'):
        if path.exists():return json.loads(path.read_text())
    return dict(name='target_validation_scout_no_improvement_v1',mode='no_improvement',metric='miou6',min_updates=1000,
                patience_evaluations=2,below_peak_margin=0.,consecutive_decline_margin=0.,require_continuing_decline=False,
                eval_period=1000,max_updates=20000,target_labels_used_for_stopping=True)


def budget(p):
    return dict(name=rb.BUDGET_NAME,phase='snd_baseline',max_updates=9000,original_policy=p,user_requested=True,
                applied_after_training_started=False,rising_cap_rule=RISING_CAP_RULE)


def curve(points):
    return [dict(iteration=i,metrics={'miou':v}) for i,v in points]


class BaselineStopTests(unittest.TestCase):
    def test_same_rule_as_pheno_arms(self):
        from scripts.run_dtst_kaggle import scientific_stop as pheno_stop
        p=policy()
        pheno=dict(name='dtst_pheno_budget_20261031_v1',phase='pheno_null',max_updates=9000,original_policy=p,user_requested=True,
                   applied_after_training_started=False,rising_cap_rule=RISING_CAP_RULE)
        cases=[[(0,.58),(1000,.585),(2000,.584),(3000,.583)],
               [(0,.58),(1000,.57),(2000,.56)],
               [(i*1000,.58+.001*i) for i in range(10)],
               [(i*1000,.58+.001*i) for i in range(10)]+[(10000,.5)],
               [(i*1000,.58+.001*(i%3)) for i in range(10)]]
        for points in cases:
            c=curve(points)
            for n in range(1,len(c)+1):
                ours,theirs=rb.scientific_stop(c[:n],p,budget(p)),pheno_stop(c[:n],p,pheno)
                self.assertEqual(ours is None,theirs is None,(points,n))
                if ours:
                    for key in ('iteration','peak_iteration','peak_miou','reason'):self.assertEqual(ours[key],theirs[key])

    def test_budget_rejects_drift(self):
        p=policy();b=budget(p)
        self.assertEqual(rb.validate_budget(b,p),9000)
        for key,value in [('max_updates',12000),('phase','pheno_null'),('rising_cap_rule',None)]:
            with self.assertRaises(ValueError):rb.validate_budget(dict(b,**{key:value}),p)
        recipe=dict(batch_size=6,base_lr=2e-5,readout_lr_mult=1,eval_period=1000,warm=dict(steps=20000,lr_max_iter=32000),
                    snd=dict(steps=12000,lr_max_iter=32000,meta=True))
        protocol=dict(recipe=recipe,source_checkpoint_sha256=rb.SOURCE_SHA,gsd_study=dict(arm='raw',target_bands='IRRG'))
        rb.validate_budget(b,p,protocol)
        for bad in (dict(protocol,dtst={}),dict(protocol,gsd_study=dict(arm='raw',target_bands='RGB')),
                    dict(protocol,recipe=dict(recipe,snd=dict(recipe['snd'],meta=False)))):
            with self.assertRaises(ValueError):rb.validate_budget(b,p,bad)

    def test_pause_trims_snd_stage_history(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            run=Path(directory);(run/'snd').mkdir()
            path=run/'snd/training_history.jsonl';path.write_text('{"iteration":1000}\n{"iteration":1001}\n')
            rb.trim_uncommitted(run,'snd',1000)
            self.assertEqual(json.loads(path.read_text())['iteration'],1000)
            self.assertIn('1001',next((run/'session_history_archive').glob('*')).read_text())


class WarmAndCommandTests(unittest.TestCase):
    def make_warm(self,root,**phenology):
        seed=root/'pheno_null/seed1';warm=seed/'warm';warm.mkdir(parents=True)
        (warm/'last.pth').write_bytes(b'weights')
        protocol=dict(protocol_signature='sig',source_checkpoint_sha256=rb.SOURCE_SHA,method='DTST_phenology',
                      phenology=dict(dict(c0=False,c1=False,fixed_delta=None),**phenology),gsd_study=dict(target_bands='IRRG'))
        (seed/'protocol.json').write_text(json.dumps(protocol))
        (warm/'stopped_early.json').write_text(json.dumps(dict(iteration=9000,protocol_signature='sig',
            checkpoint_sha256=hashlib.sha256(b'weights').hexdigest())))
        for name in ('CTR','CTR_O','meta_val','meta_val_mixup'):(warm/name).mkdir();(warm/name/'a.png').write_bytes(b'x')
        for name in ('CTR.p','CTR_O.p'):(warm/name).write_bytes(b'x')
        return warm

    def test_verify_warm(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            warm=self.make_warm(Path(directory))
            self.assertEqual(rb.verify_warm(warm)['iteration'],9000)
            (warm/'meta_val/a.png').unlink()
            with self.assertRaises(FileNotFoundError):rb.verify_warm(warm)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            with self.assertRaises(ValueError):rb.verify_warm(self.make_warm(Path(directory),c0=True))

    def test_command_is_plain_snd_stage(self):
        args=SimpleNamespace(experiment='snd_baseline',workspace=Path('/w'),settings=Path('/s'),source=Path('/src.pth'),
                             warm=Path('/warm'),data_root=Path('/d'))
        cmd=rb.command(args,Path('/o/snd_baseline/seed1'))
        self.assertEqual(Path(cmd[2]).name,'train_SND_remote.py');self.assertEqual(cmd[3],'snd')
        for flag,value in [('--readout-lr-mult','1'),('--seed','1'),('--eval-period','1000'),('--protocol','siamseg_sfuda_b2_v1')]:
            self.assertEqual(cmd[cmd.index(flag)+1],value)
        self.assertEqual(Path(cmd[cmd.index('--gsd-config')+1]).name,'raw_irrg.json')
        self.assertEqual(Path(cmd[cmd.index('--warm-checkpoint')+1]),Path('/warm/last.pth'))
        for absent in ('--pheno-config','--dtst-config','--stop-updates','--gsd-stop-updates'):self.assertNotIn(absent,cmd)

    def test_report_from_curve(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            run=Path(directory);(run/'snd').mkdir()
            rows=[dict(iteration=i,metrics=dict(miou=m,iou=[m]*5+[0.])) for i,m in [(0,.58),(1000,.6),(2000,.59)]]
            (run/'snd/evaluation_curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            rb.write_report(run,'snd')
            report=json.loads((run/'report/peak_and_endpoint.json').read_text())
            self.assertEqual(report['peak']['iteration'],1000);self.assertEqual(report['last']['iteration'],2000)
            self.assertAlmostEqual(report['peak']['miou5'],60.)


if __name__=='__main__':unittest.main()


class StableSTTests(unittest.TestCase):
    def test_stable_st_command_and_protocol(self):
        args=SimpleNamespace(experiment='stable_st',workspace=Path('/w'),settings=Path('/s'),source=Path('/src.pth'),
                             warm=None,data_root=Path('/d'))
        cmd=rb.command(args,Path('/o/stable_st/seed1'))
        self.assertEqual(Path(cmd[2]).name,'train_DTST_remote.py')
        self.assertEqual(Path(cmd[cmd.index('--config')+1]).name,'recipe_stable_st.json')
        self.assertEqual(Path(cmd[cmd.index('--dtst-config')+1]).name,'dtst_stable_st.json')
        for absent in ('--pheno-config','--warm-checkpoint','--stop-updates'):self.assertNotIn(absent,cmd)
        settings=ROOT/'kaggle_settings'
        recipe=json.loads((settings/'recipe_stable_st.json').read_text());recipe['eval_period']=1000;recipe['readout_lr_mult']=1
        dtst=json.loads((settings/'dtst_stable_st.json').read_text())
        protocol=dict(recipe=recipe,source_checkpoint_sha256=rb.SOURCE_SHA,gsd_study=dict(arm='raw',target_bands='IRRG'),dtst=dtst)
        p=policy();b=dict(budget(p),phase='stable_st')
        rb.validate_budget(b,p,protocol)
        with self.assertRaises(ValueError):rb.validate_budget(b,p,dict(protocol,dtst=dict(dtst,negative_weight=.5)))
        with self.assertRaises(ValueError):rb.validate_budget(budget(p),p,protocol)  # SND checks reject Stable-ST protocol

    def test_feedback_dispatch_and_default_unchanged(self):
        import torch
        from scripts.dtst_remote import TeacherFeedback,NFDFeedback,validate_settings
        base=json.loads((ROOT/'kaggle_settings/dtst.json').read_text()) if (ROOT/'kaggle_settings/dtst.json').exists() else \
            dict(dynamic_teacher=True,query_start=9,query_step=3,max_interval=30,points=100,ema=.99,fixed_interval=12,negative_weight=.5)
        self.assertIs(type(TeacherFeedback(validate_settings(dict(base)),'cpu')),TeacherFeedback)
        nfd=json.loads((ROOT/'kaggle_settings/dtst_stable_st.json').read_text())
        self.assertIs(type(TeacherFeedback(validate_settings(nfd),'cpu')),NFDFeedback)
        with self.assertRaises(ValueError):validate_settings(dict(nfd,proxy='other'))

    def test_nfd_feedback_cycle(self):
        import torch
        from scripts.dtst_remote import NFDFeedback
        torch.manual_seed(0)
        class Enc(torch.nn.Module):
            def __init__(self):super().__init__();self.c=torch.nn.Conv2d(3,8,4,4)
            def forward(self,x):f=self.c(x);return f,[f,f,f,f]
        class Cls(torch.nn.Module):
            def __init__(self):super().__init__();self.c=torch.nn.Conv2d(8,6,1)
            def forward(self,feats,size=None):
                z=self.c(feats[-1]);return torch.nn.functional.interpolate(z,size=size) if size else z
        enc,cls=Enc(),Cls()
        settings=json.loads((ROOT/'kaggle_settings/dtst_stable_st.json').read_text())
        fb=NFDFeedback(settings,'cpu');events=[]
        for step in range(12):
            fb.observe(enc,cls,torch.rand(6,3,32,32))
            e=fb.query(enc,cls)
            if e:events.append(e)
        # batch 6: buffer lengths 6,12,...; first query at 30 images (> max interval 21 -> forced update)
        self.assertEqual(events[0]['interval'],30);self.assertTrue(events[0]['forced'] and events[0]['update'])
        self.assertEqual(fb.updates,len(events));self.assertTrue(enc.training)
        state=fb.state_dict();other=NFDFeedback(settings,'cpu');other.load_state_dict(state)
        self.assertEqual(other.metrics(),fb.metrics())


class SNDAuthorTests(unittest.TestCase):
    def test_author_snd_command_and_protocol(self):
        args=SimpleNamespace(experiment='snd_author',workspace=Path('/w'),settings=Path('/s'),source=Path('/src.pth'),
                             warm=Path('/warm'),data_root=Path('/d'))
        cmd=rb.command(args,Path('/o/snd_author/seed1'))
        self.assertEqual(cmd[cmd.index('--snd-init')+1],'source');self.assertEqual(cmd[3],'snd')
        recipe=dict(batch_size=6,base_lr=2e-5,readout_lr_mult=1,eval_period=1000,warm=dict(steps=20000,lr_max_iter=32000),
                    snd=dict(steps=12000,lr_max_iter=32000,meta=True))
        protocol=dict(recipe=recipe,source_checkpoint_sha256=rb.SOURCE_SHA,gsd_study=dict(arm='raw',target_bands='IRRG'))
        p=policy()
        rb.validate_budget(dict(budget(p),phase='snd_author'),p,dict(protocol,snd_init='source weights + warm banks'))
        with self.assertRaises(ValueError):rb.validate_budget(dict(budget(p),phase='snd_author'),p,protocol)
        with self.assertRaises(ValueError):rb.validate_budget(budget(p),p,dict(protocol,snd_init='source'))


class SNDWarmTests(unittest.TestCase):
    def test_verify_author_snd_warm(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            seed=Path(directory)/'shared_warm/seed1';warm=seed/'warm';warm.mkdir(parents=True)
            (warm/'last.pth').write_bytes(b'w')
            proto=dict(protocol_signature='s',source_checkpoint_sha256=rb.SOURCE_SHA,method='SND',
                       gsd_study=dict(arm='raw',target_bands='IRRG'),recipe=dict(warm=dict(ema=.99,ema_interval=12,meta=False)))
            (seed/'protocol.json').write_text(json.dumps(proto))
            (warm/'complete.json').write_text(json.dumps(dict(iteration=20000,protocol_signature='s',checkpoint_sha256=hashlib.sha256(b'w').hexdigest())))
            for n in ('CTR','CTR_O','meta_val','meta_val_mixup'):(warm/n).mkdir();(warm/n/'a.png').write_bytes(b'x')
            self.assertEqual(rb.verify_warm(warm,'snd_warm')['iteration'],20000)
            proto['method']='DTST_B2';(seed/'protocol.json').write_text(json.dumps(proto))
            with self.assertRaises(ValueError):rb.verify_warm(warm,'snd_warm')
