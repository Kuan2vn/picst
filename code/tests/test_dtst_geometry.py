import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts.dtst_geometry import donor_scale, masked_context_loss, validate_geometry
from scripts.dtst_pilot_budget import validate_budget, cap_decision, RISING_CAP_RULE


BASE = dict(name='dtst_geometry_20261016_v1', target_gsd_m=.05, donor_mode='baseline', context=None)
CONTEXT = dict(kind='mic_style', weight=.1, mask_ratio=.5, block_size=8, confidence_threshold=.968)


class GeometryTests(unittest.TestCase):
    def test_scales_and_baseline_rng_contract(self):
        self.assertEqual(donor_scale('baseline', 1.5, .2), 1.3)
        self.assertEqual(donor_scale('independent', 1.5, .2), 1.2)
        self.assertEqual(donor_scale('coupled', 1.5, .2), 1.5)
        with self.assertRaises(ValueError):
            donor_scale('coupled', None, .2)
        with self.assertRaises(ValueError):
            validate_geometry(dict(BASE, donor_mode='unknown'))
        with self.assertRaises(ValueError):
            validate_geometry(dict(BASE, context=dict(CONTEXT, weight=-1)))

    def test_real_mixer_coupling_and_size_skip(self):
        from scripts.snd_remote import ObjectMixer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'CTR').mkdir()
            label = np.full((64,64),255,np.uint8);label[10:20,12:22]=4
            Image.fromarray(label).save(root/'CTR'/'donor.png')
            runtime = SimpleNamespace(output=root, geometry_settings=dict(BASE, donor_mode='coupled'),
                                      image=lambda name:Image.new('RGB',(64,64),(100,120,140)))
            mixer = ObjectMixer(runtime)
            _, pasted, stats = mixer.mix(Image.new('RGB',(64,64)), [4], [1.], 1.5, True)
            self.assertEqual(stats['accepted'][4],1)
            self.assertEqual(int((pasted==4).sum()),225)
            self.assertAlmostEqual(stats['pitch_ratio_sum'][4],1.)
            _, _, stats = mixer.mix(Image.new('RGB',(12,12)), [4], [1.], 1.5, True)
            self.assertEqual(stats['skipped_size'][4],1)

    def test_baseline_mixer_preserves_pixels_labels_and_rng(self):
        from scripts.snd_remote import ObjectMixer
        import random
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'CTR').mkdir()
            label=np.full((64,64),255,np.uint8);label[10:20,12:22]=4
            Image.fromarray(label).save(root/'CTR'/'donor.png')
            runtime=SimpleNamespace(output=root,image=lambda name:Image.new('RGB',(64,64),(100,120,140)))
            mixer=ObjectMixer(runtime);image=Image.new('RGB',(64,64))
            np.random.seed(17);random.seed(17)
            a,b=mixer.mix(image,[4],[1.]);rng_a=np.random.get_state();py_a=random.getstate()
            runtime.geometry_settings=BASE
            np.random.seed(17);random.seed(17)
            c,d,stats=mixer.mix(image,[4],[1.],1.5,True);rng_b=np.random.get_state()
            np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
            np.testing.assert_array_equal(rng_a[1],rng_b[1]);self.assertEqual(rng_a[2:],rng_b[2:])
            self.assertEqual(py_a,random.getstate())

    def test_context_backward_does_not_train_teacher_and_ignores_invalid(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layer=torch.nn.Conv2d(3,4,1)
            def forward(self,x):return None,self.layer(x)
        class Classifier(torch.nn.Conv2d):
            def forward(self,x,size):return super().forward(x)
        encoder=Encoder();classifier=Classifier(4,6,1)
        images=torch.randn(2,3,16,16)
        teacher=torch.zeros(2,6,16,16);teacher[:,2]=.99;teacher[:,1]=.01;teacher.requires_grad_()
        valid=torch.ones(2,16,16,dtype=torch.bool);valid[:,:3]=False
        loss,record=masked_context_loss(encoder,classifier,images,teacher,valid,CONTEXT)
        loss.backward()
        self.assertGreater(float(encoder.layer.weight.grad.abs().sum()),0)
        self.assertGreater(float(classifier.weight.grad.abs().sum()),0)
        self.assertIsNone(teacher.grad)
        self.assertGreater(record['context_mask_fraction'],0)
        self.assertLess(record['context_mask_fraction'],1)
        zero,_=masked_context_loss(encoder,classifier,images,teacher,torch.zeros_like(valid),CONTEXT)
        self.assertEqual(float(zero.detach()),0)

    def test_new_budget_uses_existing_peak_stop_and_cap_extension(self):
        policy=json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping_scout_2eval.json').read_text())
        budget=dict(name='dtst_geometry_budget_20261016_v1',phase='coupled_scale',max_updates=9000,
                    original_policy=policy,user_requested=True,applied_after_training_started=False,
                    rising_cap_rule=RISING_CAP_RULE)
        self.assertEqual(validate_budget(budget,policy),9000)
        curve=[dict(iteration=i,metrics={'miou':.5+i/100000}) for i in range(0,10000,1000)]
        self.assertIsNone(cap_decision(curve,budget,policy))
        curve.append(dict(iteration=10000,metrics={'miou':.58}))
        self.assertTrue(cap_decision(curve,budget,policy)['pilot_cap_extension'])
        with self.assertRaises(ValueError):validate_budget(dict(budget,max_updates=6000),policy)


if __name__=='__main__':unittest.main()
