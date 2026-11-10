"""Integration gates for the remote SND adapter, using CPU synthetic images."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from core.utils.losses import full2weak
from datasets.generate_city_label_info import gen_lb_info
from scripts.snd_remote import SNDRemoteRuntime, align_full, ObjectMixer, TargetTraining
from train_SND_remote import portable
from train_SRMP_remote import rng_state


class Encoder(nn.Module):
    def forward(self, image):
        return image, image


class Classifier(nn.Module):
    def forward(self, features, size):
        # Six repeated vertical classes with spatial confidence variation.
        h, w = size
        x = torch.arange(w, device=features.device)
        labels = x*6//w
        logits = F.one_hot(labels, 6).float().T[None, :, None].expand(len(features), 6, h, w)
        return logits * (2 + x.float()[None, None, None]/w)


def read_mixing_probability(dataset, connection):
    connection.send('ready')
    connection.recv()
    connection.send(dataset.mix_p.tolist())
    connection.close()


class SNDAdapterTest(unittest.TestCase):
    def test_curve_peak_preserves_rng_and_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = SNDRemoteRuntime.__new__(SNDRemoteRuntime)
            runtime.root = runtime.output = Path(temporary)
            runtime.device, runtime.workers, runtime.signature = torch.device('cpu'), 0, 'test'
            runtime.architecture = 'resnet101_aspp'
            runtime.eval_split = 'target_eval'
            runtime.eval_period, runtime.eval_limit, runtime.stop_iteration = 500, 0, 1100
            runtime.evaluated_iterations = set()
            encoder, classifier = Encoder().eval(), Classifier().train()
            before = rng_state()
            scores = iter([(.2, .4), (.5, .35), (.3, .6), (.4, .55)])
            def evaluate(*args):
                self.assertFalse(encoder.training)
                self.assertFalse(classifier.training)
                torch.rand(3); np.random.rand(3)
                score, score5 = next(scores)
                return {'miou': score, 'miou5': score5, 'class_order': ['a'],
                        'overall_accuracy': score, 'mean_accuracy': score, 'mean_f1': score,
                        'mean_precision': score, 'mean_recall': score,
                        **{key + '_by_class': {'a': score} for key in
                           ('iou', 'accuracy', 'f1', 'precision', 'recall')}}
            with patch('scripts.diagnose_crots_remote.EvaluationImages', return_value=[1]), \
                 patch('scripts.diagnose_crots_remote.evaluate_at_size', side_effect=evaluate) as observed:
                with self.assertLogs(level='INFO') as logs:
                    for step in (0, 250, 500, 500, 1000, 1100):
                        if step == 1100:
                            (runtime.output/'observed_peak_miou5.json').unlink()
                        runtime.record_evaluation(step, encoder, classifier)
                self.assertEqual(observed.call_count, 4)
            self.assertIn('previous peak mIoU6 none', logs.output[0])
            self.assertIn('previous peak mIoU6 20.00% at step 0', logs.output[1])
            self.assertIn('previous peak mIoU6 50.00% at step 500', logs.output[2])
            self.assertIn('previous peak mIoU6 50.00% at step 500', logs.output[3])
            self.assertIn('previous peak mIoU5 none', logs.output[0])
            self.assertIn('previous peak mIoU5 40.00% at step 0', logs.output[1])
            self.assertIn('previous peak mIoU5 40.00% at step 0', logs.output[2])
            self.assertIn('previous peak mIoU5 60.00% at step 1000', logs.output[3])
            self.assertIn('peak mIoU6 50.00% at step 500', logs.output[3])
            self.assertIn('peak mIoU5 60.00% at step 1000', logs.output[3])
            points = [json.loads(line) for line in (runtime.output/'evaluation_curve.jsonl').read_text().splitlines()]
            self.assertEqual([p['iteration'] for p in points], [0, 500, 1000, 1100])
            peak = json.loads((runtime.output/'observed_peak.json').read_text())
            self.assertEqual(peak['iteration'], 500)
            self.assertEqual(peak['metrics']['miou'], .5)
            peak5 = json.loads((runtime.output/'observed_peak_miou5.json').read_text())
            self.assertEqual(peak5['iteration'], 1000)
            self.assertEqual(peak5['metrics']['miou5'], .6)
            self.assertFalse(peak['used_for_training_or_stopping'])
            after = rng_state()
            self.assertTrue(torch.equal(before['torch'], after['torch']))
            np.testing.assert_array_equal(before['numpy'][1], after['numpy'][1])
            self.assertFalse(encoder.training)
            self.assertTrue(classifier.training)

    def test_mixing_probability_updates_reach_worker(self):
        context = torch.multiprocessing.get_context('spawn')
        dataset = TargetTraining.__new__(TargetTraining)
        dataset._mix_p = torch.full((6,), 1/6, dtype=torch.float64).share_memory_()
        parent, child = context.Pipe()
        process = context.Process(target=read_mixing_probability, args=(dataset, child))
        process.start()
        try:
            self.assertTrue(parent.poll(30))
            self.assertEqual(parent.recv(), 'ready')
            dataset.mix_p = [0, 0, 0, 0, 1, 0]
            parent.send(True)
            self.assertTrue(parent.poll(30))
            self.assertEqual(parent.recv(), [0, 0, 0, 0, 1, 0])
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate(); process.join()
            parent.close(); child.close()

    def test_alignment_matches_native_without_padding(self):
        features = torch.arange(2*3*64*64).float().reshape(2, 3, 64, 64)
        params = default_collate([
            {'rescale_size': (64, 64), 'padding': (0, 0, 0, 0), 'random_crop_axis': (8, 56, 0, 48), 'RandomHorizontalFlip': False},
            {'rescale_size': (80, 80), 'padding': (0, 0, 0, 0), 'random_crop_axis': (16, 64, 24, 72), 'RandomHorizontalFlip': True},
        ])
        for ratio in (1, 8):
            self.assertTrue(torch.equal(full2weak(features, params, down_ratio=ratio), align_full(features, params, down_ratio=ratio)))

    def test_padding_and_pseudo_ignore_align(self):
        features = torch.arange(1, 17).float().reshape(1, 1, 4, 4)
        params = default_collate([{'rescale_size': (4, 4), 'padding': (2, 2, 2, 2),
                                  'random_crop_axis': (1, 7, 0, 6), 'RandomHorizontalFlip': True}])
        expected = F.pad(features, (2, 2, 2, 2))[:, :, 1:7, 0:6].flip(-1)
        actual = align_full(features, params, nearest=True)
        self.assertTrue(torch.equal(actual, expected))
        params['valid'] = expected[:, 0] > 0
        labels = SNDRemoteRuntime.mask_padding(torch.ones((1, 6, 6), dtype=torch.long), params)
        self.assertTrue(torch.equal(labels == 255, expected[:, 0] == 0))

    def test_candidate_refresh_meta_and_mixer_use_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'manifests').mkdir()
            (root/'images').mkdir()
            output = root/'run'
            output.mkdir()
            rows = []
            for i in range(4):
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                image[:, :, i % 3] = 80+i*20
                Image.fromarray(image).save(root/'images'/f'{i}.png')
                rows.append({'image': f'images/{i}.png', 'label': ''})
            manifest = root/'manifests/isprs_pd_rgb_train.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image', 'label'])
                writer.writeheader(); writer.writerows(rows)
            runtime = SNDRemoteRuntime(root, output, 'cpu', size=64, workers=0)
            cfg = SimpleNamespace(TCR=SimpleNamespace(TOPK_CANDIDATE=.5),
                                  METAPL=SimpleNamespace(TOP_K=.25), MODEL=SimpleNamespace(NUM_CLASSES=6), OUTPUT_DIR=str(output))
            encoder, classifier = Encoder(), Classifier()
            runtime.run_candidate(cfg, encoder, classifier, 0, False, init_candidate=True, update_meta=True)
            self.assertTrue(encoder.training)
            for name in ('CTR', 'CTR_O', 'meta_val', 'meta_val_mixup'):
                gen_lb_info(cfg, name)
                self.assertGreater(len(list((output/name).glob('*.png'))), 0)
            old = np.asarray(Image.open(output/'CTR_O/0.png'))
            runtime.run_candidate(cfg, encoder, classifier, 0, False, update_meta=False)
            np.testing.assert_array_equal(old, np.asarray(Image.open(output/'CTR_O/0.png')))
            records = [json.loads(line) for line in (output/'candidates.jsonl').read_text().splitlines()]
            self.assertFalse(records[-1]['initial'])
            self.assertGreater(min(records[-1]['thresholds']), .5)
            images, labels = runtime.sample_val_images(cfg, torch.tensor(old[None].copy()).long())
            self.assertEqual(tuple(images.shape), (1, 3, 64, 64))
            self.assertTrue(set(labels.unique().tolist()) <= set(range(6)) | {255})
            mixer = ObjectMixer(runtime)
            for c in range(6):
                _, pasted = mixer.mix(np.zeros((128, 128, 3), np.uint8), [c], [1.])
                self.assertTrue(set(np.unique(pasted)) <= {c, 255})
            # Source/target GT cannot accidentally enter the training adapter.
            rows[0]['label'] = 'images/0.png'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image', 'label'])
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'label visibility'):
                SNDRemoteRuntime(root, output, 'cpu')

    def test_portable_contains_no_model_or_pseudo_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('protocol.json', 'run.log', 'last.pth', 'CTR.p', 'pseudo.png'):
                (root/name).write_text('{}')
            portable(root)
            with zipfile.ZipFile(root/'results_portable.zip') as archive:
                self.assertEqual(set(archive.namelist()), {'protocol.json', 'run.log'})


if __name__ == '__main__':
    unittest.main()
