import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from core.utils.losses import BinaryCrossEntropy
from scripts.dtst_remote import TeacherFeedback, negative_loss, probe_probability, validate_settings
from scripts.gsd_reliability import selective_outer_loss
from scripts.dtst_study_report import rows


SETTINGS = dict(dynamic_teacher=True, query_start=1, query_step=1, max_interval=2,
                points=8, fixed_interval=12, ema=.99, negative_weight=.5)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, images):
        return images, self.bn(images)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 1)

    def forward(self, features, size):
        return self.conv(features)


class DTSTTests(unittest.TestCase):
    def test_negative_matches_released_loss_and_gradients(self):
        logits = torch.randn(2, 6, 3, 4, requires_grad=True)
        probability = torch.randn_like(logits).softmax(1)
        valid = torch.ones(2, 3, 4, dtype=torch.bool)
        labels = torch.where(probability <= 1/6, 0., 255.).reshape(12, 3, 4)
        expected = .5 * BinaryCrossEntropy()(logits.reshape(12, 1, 3, 4), labels)
        actual = negative_loss(logits, probability, valid)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, logits)[0])

    def test_padding_and_empty_negative_support(self):
        logits = torch.randn(1, 6, 2, 3, requires_grad=True)
        probability = logits.detach().softmax(1)
        valid = torch.tensor([[[True, False, True], [True, False, True]]])
        loss = negative_loss(logits, probability, valid)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[:, :, :, 1], torch.zeros_like(logits.grad[:, :, :, 1])))
        empty = negative_loss(logits, probability, torch.zeros_like(valid))
        self.assertEqual(float(empty.detach()), 0)
        self.assertTrue(empty.requires_grad)

    def test_probe_preserves_individual_modes_and_bn(self):
        encoder, classifier = Encoder().train(), Classifier().train()
        encoder.bn.eval()
        modes = [m.training for model in (encoder,classifier) for m in model.modules()]
        mean = encoder.bn.running_mean.clone()
        probe_probability(encoder, classifier, torch.randn(2,3,4,4))
        self.assertEqual(modes, [m.training for model in (encoder,classifier) for m in model.modules()])
        torch.testing.assert_close(mean, encoder.bn.running_mean)

    def test_gain_direction_cap_and_resume(self):
        encoder, classifier = Encoder(), Classifier()
        feedback = TeacherFeedback(SETTINGS, 'cpu')
        images = torch.randn(1,3,4,4)
        with patch('scripts.dtst_remote.density', return_value=1.):
            for _ in range(2):
                feedback.observe(encoder,classifier,images)
                self.assertFalse(feedback.query(encoder,classifier)['update'])
            stream = io.BytesIO()
            torch.save(feedback.state_dict(), stream); stream.seek(0)
            restored = TeacherFeedback(SETTINGS, 'cpu')
            restored.load_state_dict(torch.load(stream, weights_only=False))
            restored.observe(encoder,classifier,images)
            event = restored.query(encoder,classifier)
            self.assertTrue(event['forced']); self.assertTrue(event['update'])
            self.assertEqual(restored.updates,1); self.assertEqual(restored.buffer,[])
        with patch('scripts.dtst_remote.density', return_value=1.):
            feedback = TeacherFeedback(SETTINGS,'cpu'); feedback.observe(encoder,classifier,images)
        with patch('scripts.dtst_remote.density', return_value=2.):
            self.assertTrue(feedback.query(encoder,classifier)['update'])

    def test_selector_q1_keeps_full_dtst_objective(self):
        logits = torch.randn(1,6,3,4,requires_grad=True)
        probability = torch.randn_like(logits).softmax(1)
        labels = probability.argmax(1); labels[:,0,0]=255
        pasted = torch.full_like(labels,255)
        ce=F.cross_entropy(logits,labels,ignore_index=255,reduction='none')
        neg=negative_loss(logits,probability,torch.ones_like(labels,dtype=torch.bool))
        selected,_,_=selective_outer_loss(ce,torch.ones_like(ce),labels,pasted,labels!=255,probability.max(1).values,1)
        torch.testing.assert_close(selected+neg,ce.mean()+neg,atol=0,rtol=0)

    def test_settings_reject_invalid_policy(self):
        validate_settings(SETTINGS)
        with self.assertRaises(ValueError): validate_settings(dict(SETTINGS,query_start=3,max_interval=2))
        with self.assertRaises(ValueError): validate_settings(dict(SETTINGS,negative_weight=float('nan')))

    def test_report_accepts_only_a_partial_final_line(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'history.jsonl'
            path.write_text('{"iteration":20}\n{"iteration":')
            self.assertEqual(rows(path),[{'iteration':20}])
            path.write_text('broken\n{"iteration":20}\n')
            with self.assertRaises(json.JSONDecodeError): rows(path)

    def test_short_dtst_endpoint_needs_no_target_selected_checkpoint(self):
        from train_SND_remote import execute_stage
        with TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source.pth'; source.write_bytes(b'source')
            recipe=json.loads(Path('configs/snd_vh2pd_b2.json').read_text())
            args=SimpleNamespace(output=root/'run',data_root=root,source_checkpoint=source,protocol=recipe['protocol'],
                dtst_settings=SETTINGS,gsd_stop_updates=2,warm_stop_updates=None,extend_warm_to=None,
                device='cpu',target_limit=4,workers=0,smoke_only=False,seed=1,eval_period=0,eval_limit=0,
                aux_settings=None,gsd_settings=None)
            runtime=MagicMock(); runtime.device=torch.device('cpu'); runtime.aux=None
            runtime.teacher_inferences=runtime.bank_refreshes=runtime.meta_images=0
            runtime.aux_inherited_forward_count=0
            runtime.checkpoint.side_effect=lambda *unused:(args.output/'warm/last.pth').write_bytes(b'model')
            def train(cfg,*unused,**kwargs):
                runtime.last_iteration=cfg.SOLVER.STOP_ITER
                return Encoder(),Classifier()
            with patch('train_SND_remote.SNDRemoteRuntime',return_value=runtime),\
                 patch('train_SND_remote.train',side_effect=train),\
                 patch('train_SND_remote.finalize_aux_warm',side_effect=AssertionError('No target selection')):
                execute_stage('warm',recipe,args,'test')
            self.assertEqual(json.loads((args.output/'warm/complete.json').read_text())['iteration'],2)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
