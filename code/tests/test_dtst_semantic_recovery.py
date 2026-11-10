import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F
from scripts.dtst_semantic_recovery import SeedMemory, SemanticRecovery, validate_recovery, name_id
from scripts.dtst_remote import negative_loss
from scripts.dtst_pilot_budget import validate_budget, cap_decision, RISING_CAP_RULE

CONFIG=dict(name='dtst_semantic_recovery_20261016_v1',features='ema',grid=32,
            seed_confidence=.9,vote_confidence=.8,min_cosine=.65,correction_weight=.5,
            max_native_confidence=.95,memory_per_class=128,seeds_per_image_class=8,
            min_seeds=8,neighbors=5,required_votes=4,max_age=2000)


class RecoveryTests(unittest.TestCase):
    def memory(self):
        m=SeedMemory(CONFIG)
        f=torch.tensor([[1.,0.]]*16+[[0.,1.]]*16)
        p=F.one_hot(torch.tensor([3]*16+[4]*16),6).float()
        ids=torch.tensor([11]*8+[12]*8+[21]*8+[22]*8)
        m.add(f,p,ids,1)
        return m

    def test_query_class_vote_and_excludes_same_image(self):
        m=self.memory()
        label,valid=m.query(torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([90,91]),2)
        self.assertEqual(label.tolist(),[3,4]);self.assertTrue(valid.all())
        # Only one supporting other image remains; reject self-derived evidence.
        _,valid=m.query(torch.tensor([[1.,0.]]),torch.tensor([11]),2)
        self.assertFalse(valid.any())

    def test_empty_stale_and_mixed_neighborhood_abstain(self):
        m=SeedMemory(CONFIG)
        _,valid=m.query(torch.tensor([[1.,0.]]),torch.tensor([1]),1)
        self.assertFalse(valid.any())
        m=self.memory()
        _,valid=m.query(torch.tensor([[1.,0.]]),torch.tensor([90]),3000)
        self.assertFalse(valid.any())
        m.labels=torch.tensor([3,4]*8+[4]*16)
        _,valid=m.query(torch.tensor([[1.,0.]]),torch.tensor([90]),2)
        self.assertFalse(valid.any())

    def test_resume_identity_and_bounded_memory(self):
        m=self.memory();restored=SeedMemory(CONFIG)
        restored.load_state_dict(m.state_dict(),'cpu')
        q=torch.tensor([[1.,0.],[0.,1.]])
        a=m.query(q,torch.tensor([90,91]),2)
        b=restored.query(q,torch.tensor([90,91]),2)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
        for step in range(2,100):
            m.add(torch.tensor([[1.,0.]]*16),F.one_hot(torch.tensor([3]*16),6).float(),
                  torch.tensor([step*100]*8+[step*100+1]*8),step)
        self.assertLessEqual(int((m.labels==3).sum()),CONFIG['memory_per_class'])
        self.assertEqual(name_id('a'),name_id('a'));self.assertNotEqual(name_id('a'),name_id('b'))

    def test_recipient_only_padding_and_confidence_guard(self):
        recovery=SemanticRecovery(SimpleNamespace(device='cpu'),CONFIG)
        recovery.candidate=torch.full((1,2,3),3)
        recovery.accepted=torch.ones((1,2,3),dtype=torch.bool)
        labels=torch.tensor([[[2,2,255],[2,2,2]]]);donor=torch.full_like(labels,255);donor[:,0,0]=2
        valid=torch.ones_like(labels,dtype=torch.bool);valid[:,1,0]=False
        p=torch.ones(1,6,2,3)/6;p[:,:,1,1]=0;p[:,2,1,1]=.99
        out=recovery.apply(labels,donor,p,valid)
        self.assertEqual(out.tolist(),[[[2,3,3],[2,2,3]]])
        self.assertEqual(recovery.last_record['recovery_filled_ignore_by_class'][3],1)
        self.assertEqual(recovery.protected[0,0,0],255)

    def test_correction_loss_and_negative_protection_gradients(self):
        logits=torch.randn(1,6,2,3,requires_grad=True)
        p=torch.full_like(logits,.01);p[:,2]=.95
        valid=torch.ones(1,2,3,dtype=torch.bool)
        protected=torch.full((1,2,3),255);protected[:,0,0]=3
        loss=negative_loss(logits,p,valid,protected_labels=protected);loss.backward()
        self.assertEqual(float(logits.grad[0,3,0,0]),0)
        self.assertGreater(float(logits.grad[0,3,0,1]),0)
        torch.testing.assert_close(negative_loss(logits,p,valid),
            negative_loss(logits,p,valid,protected_labels=torch.full_like(protected,255)),atol=0,rtol=0)
        recovery=SemanticRecovery(SimpleNamespace(device='cpu'),CONFIG)
        recovery.corrected=torch.tensor([[[True,False,False],[False,False,False]]])
        self.assertAlmostEqual(float(recovery.positive_loss(torch.ones(1,2,3),torch.ones(1,2,3))),5.5/6,places=6)

    def test_prepare_alignment_and_teacher_stop_gradient(self):
        runtime=SimpleNamespace(device='cpu',full2weak=lambda x,p,nearest=False:x)
        recovery=SemanticRecovery(runtime,CONFIG)
        native=torch.randn(2,6,32,32,requires_grad=True)
        features=(torch.randn(2,3,16,16,requires_grad=True),torch.randn(2,4,8,8,requires_grad=True))
        recovery.prepare(torch.randn(2,3,32,32),features,native,['a','b'],{},1)
        self.assertEqual(tuple(recovery.candidate.shape),(2,32,32))
        self.assertFalse(recovery.accepted.requires_grad)
        self.assertFalse(recovery.memory.features.requires_grad)

    def test_validation_and_new_cap_rule(self):
        with self.assertRaises(ValueError):validate_recovery(dict(CONFIG,features='unknown'))
        with self.assertRaises(ValueError):validate_recovery(dict(CONFIG,neighbors=2,required_votes=4))
        policy=json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping_scout_2eval.json').read_text())
        budget=dict(name='dtst_recovery_budget_20261016_v1',phase='dino_recovery',max_updates=9000,
                    original_policy=policy,user_requested=True,applied_after_training_started=False,
                    rising_cap_rule=RISING_CAP_RULE)
        self.assertEqual(validate_budget(budget,policy),9000)
        curve=[dict(iteration=i,metrics={'miou':.5+i/100000}) for i in range(0,10000,1000)]
        self.assertIsNone(cap_decision(curve,budget,policy))
        curve.append(dict(iteration=10000,metrics={'miou':.58}))
        self.assertTrue(cap_decision(curve,budget,policy)['pilot_cap_extension'])


if __name__=='__main__':unittest.main()
