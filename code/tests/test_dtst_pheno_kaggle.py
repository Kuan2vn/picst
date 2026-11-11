"""Phenology arms on the Kaggle runner keep the registered HPC budget and rising-cap rule, without a GPU."""
import json,unittest
from pathlib import Path
from scripts.run_dtst_kaggle import PHASES,PHENO,scientific_stop
from scripts.dtst_pilot_budget import validate_budget

ROOT=Path(__file__).parents[1]
ARMS={'pheno_c0':(True,False),'pheno_c0_c1':(True,True),'pheno_c1':(False,True),'pheno_null':(False,False),
      'pheno_c0multi':(True,False),'pheno_c0multi_c1plaus':(True,True),'pheno_c0_c1plaus':(True,True)}
PROTECT={'pheno_c0':False,'pheno_c0_c1':True,'pheno_c1':True,'pheno_null':False,
         'pheno_c0multi':False,'pheno_c0multi_c1plaus':True,'pheno_c0_c1plaus':True}

def budget(phase,policy,**change):
    return dict(dict(name='dtst_pheno_budget_20261031_v1',phase=phase,max_updates=9000,original_policy=policy,
        user_requested=True,applied_after_training_started=False,
        rising_cap_rule='new_peak_at_cap_then_first_strict_decline_v1'),**change)

def settings(arm):
    return json.loads((ROOT/'kaggle_settings'/(arm+'.json')).read_text())

class PhenoKaggleTests(unittest.TestCase):
    def setUp(self):
        self.policy=json.loads((ROOT/'kaggle_settings/early_stopping_scout_2eval.json').read_text())
    def test_arms_settings_and_budget(self):
        self.assertTrue(set(ARMS)<=set(PHENO));self.assertTrue(set(PHENO)<=set(PHASES))
        for arm,flags in ARMS.items():
            s=settings(arm)
            self.assertEqual((s['c0'],s['c1']),flags)
            self.assertEqual(s['protect_pasted_tree'],PROTECT[arm])
            self.assertEqual(validate_budget(budget(arm,self.policy),self.policy),9000)
        with self.assertRaises(ValueError):validate_budget(budget('pheno_c1',self.policy,max_updates=12000),self.policy)
        with self.assertRaises(ValueError):validate_budget(budget('raw',self.policy),self.policy)
    def test_ablations_differ_from_c0_c1_only_where_declared(self):
        base=settings('pheno_c0_c1')
        diff=lambda arm:{k for k in set(base)|set(settings(arm)) if base.get(k)!=settings(arm).get(k)}
        self.assertEqual({k for k in base if base[k]!=settings('pheno_c0')[k]},{'c1','protect_pasted_tree'})
    def test_patience_and_rising_cap(self):
        b=budget('pheno_c1',self.policy)
        curve=[dict(iteration=i,metrics={'miou':v}) for i,v in [(0,.52),(1000,.55),(2000,.56),(3000,.559),(4000,.558)]]
        self.assertEqual(scientific_stop(curve,self.policy,b)['iteration'],4000)
        curve=[dict(iteration=i*1000,metrics={'miou':.5+i/100}) for i in range(10)]
        self.assertIsNone(scientific_stop(curve,self.policy,b))
        curve.append(dict(iteration=10000,metrics={'miou':.5}))
        self.assertEqual(scientific_stop(curve,self.policy,b)['iteration'],10000)

try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'needs torch')
class PhenoOperatorTests(unittest.TestCase):
    def setUp(self):
        from scripts import dtst_phenology as p
        self.p=p;torch.manual_seed(0)
        self.x=torch.rand(2,3,64,64)*255;self.tree=torch.zeros(2,64,64,dtype=torch.bool);self.tree[:,16:48,16:48]=True
        self.ground=~self.tree;self.a=torch.full((2,1,1,1),.5)
    def test_validation(self):
        for arm in ARMS:self.p.validate_phenology(settings(arm))
        with self.assertRaises(ValueError):self.p.validate_phenology(dict(settings('pheno_c0_c1'),c1_operator='jitter'))
        with self.assertRaises(ValueError):self.p.validate_phenology(dict(settings('pheno_c0_c1'),extra=1))
    def test_default_operator_is_the_hpc_formula(self):
        edited,ok=self.p.leaf_off(self.x,self.tree,self.ground,self.a)
        mu_g,_=self.p._local_mean(self.x,self.ground,65);mu_t,_=self.p._local_mean(self.x,self.tree,65)
        self.assertTrue(torch.equal(edited[ok[:,None].expand_as(edited)],(mu_g+self.a*(self.x-mu_t))[ok[:,None].expand_as(edited)]))
    def test_random_direction_keeps_shift_magnitude(self):
        u=torch.nn.functional.normalize(torch.randn(2,3,1,1),dim=1)
        e0,ok0=self.p.leaf_off(self.x,self.tree,self.ground,self.a)
        e1,ok1=self.p.leaf_off(self.x,self.tree,self.ground,self.a,direction=u)
        self.assertTrue(torch.equal(ok0,ok1))
        mu_t,_=self.p._local_mean(self.x,self.tree,65)
        texture=self.a*(self.x-mu_t)
        n0=(e0-texture-mu_t).norm(dim=1)[ok0];n1=(e1-texture-mu_t).norm(dim=1)[ok1]
        self.assertTrue(torch.allclose(n0,n1,rtol=1e-4,atol=1e-3))
        self.assertFalse(torch.allclose(e0,e1))
    def test_rgb_target_only_changes_bands(self):
        irrg=json.loads((ROOT/'kaggle_settings/raw_irrg.json').read_text());rgb=json.loads((ROOT/'kaggle_settings/raw_rgb.json').read_text())
        self.assertEqual(irrg['target_bands'],'IRRG');self.assertEqual(rgb,dict(irrg,target_bands='RGB'))
        src=(ROOT/'scripts/run_dtst_kaggle.py').read_text()
        self.assertIn("'raw_rgb.json' if args.target_bands=='RGB' else 'raw_irrg.json'",src)
        self.assertIn("p.add_argument('--target-bands',choices=('IRRG','RGB'),default='IRRG')",src)

if __name__=='__main__':unittest.main()
