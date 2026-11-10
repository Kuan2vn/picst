"""CPU regression: source selection, LR groups, and monitoring independence."""
import csv
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn

import train_SRMP_remote as trainer
from scripts.prepare_dgle_remote_sensing import VH_TRAIN, VH_VAL
from scripts.snd_remote import SNDRemoteRuntime


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        x = self.conv(x)
        return x, x


class Classifier(nn.Conv2d):
    def __init__(self):
        super().__init__(4, 6, 1)

    def forward(self, x, size):
        return super().forward(x)


class SourceTest(unittest.TestCase):
    def test_source_selection_and_monitor_does_not_change_training(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'manifests').mkdir()
            for name, tiles in [('isprs_vh_irrg_train', VH_TRAIN), ('isprs_vh_irrg_val', VH_VAL),
                                ('isprs_pd_rgb_eval', (99,))]:
                (root/name).mkdir()
                rows = []
                for tile in tiles:
                    pixels = np.arange(16*16*3, dtype=np.uint8).reshape(16, 16, 3)
                    mask = np.tile(np.arange(16, dtype=np.uint8) % 6, (16, 1))
                    image, label = f'{name}/{tile}_0_0.png', f'{name}/{tile}_label.png'
                    Image.fromarray(pixels).save(root/image)
                    Image.fromarray(mask).save(root/label)
                    rows.append((image, label))
                with (root/'manifests'/f'{name}.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream); writer.writerow(['image', 'label']); writer.writerows(rows)
            protocols = {'files_sha256': {'manifest': 'train', 'initial_backbone': 'imagenet',
                                         'source_val_manifest': 'validation'},
                         'selection_rule': 'best source validation'}
            args = SimpleNamespace(task='vh2pd', data_root=root, batch_size=4, accum_steps=1,
                                   crop_size=16, train_policy='siamseg_isprs', workers=0, seed=1,
                                   lr=.001, head_lr_mult=10., amp=False, steps=2, log_period=1,
                                   save_period=2, backbone_weights=root/'unused')
            outputs = []
            real_sgd = torch.optim.SGD
            optimizers = []

            def optimizer(*a, **kw):
                optimizers.append(real_sgd(*a, **kw))
                return optimizers[-1]

            for monitor in (False, True):
                random.seed(1); np.random.seed(1); torch.manual_seed(1)
                args.output_dir = root/str(monitor); args.output_dir.mkdir()
                args.source_val_split = 'isprs_vh_irrg_val' if monitor else None
                args.eval_period = int(monitor)
                (args.output_dir/'protocol_source.json').write_text(json.dumps(protocols))
                scores = iter([.1, .1, .8, .4, .2, .9])  # val / target at steps 0, 1, 2

                def evaluate(encoder, classifier, dataset, eval_args, device):
                    self.assertFalse(eval_args.amp)
                    encoder.eval(); classifier.eval()
                    random.random(); np.random.rand(); torch.rand(3)
                    return {'miou': next(scores)}

                with patch.object(trainer, 'make_model', side_effect=lambda _: (Encoder(), Classifier())), \
                     patch.object(trainer, 'initialize_backbone'), \
                     patch.object(trainer, 'evaluate', side_effect=evaluate), \
                     patch.object(torch.optim, 'SGD', side_effect=optimizer):
                    trainer.source_train(args, trainer.TASKS['vh2pd'], torch.device('cpu'))
                outputs.append(trainer.load_checkpoint(args.output_dir/'source_last.pth'))
                self.assertAlmostEqual(optimizers[-1].param_groups[1]['lr'],
                                       10 * optimizers[-1].param_groups[0]['lr'])
            for part in ('feature_extractor', 'classifier'):
                for key in outputs[0][part]:
                    self.assertTrue(torch.equal(outputs[0][part][key], outputs[1][part][key]), key)
            report = json.loads((args.output_dir/'source_training_report.json').read_text())
            self.assertTrue(report['completed'])
            self.assertEqual(report['source_best']['iteration'], 1)
            self.assertEqual(report['source_best']['target_eval']['miou'], .4)
            self.assertEqual(report['observed_peaks']['target_eval']['iteration'], 2)
            best = trainer.load_checkpoint(args.output_dir/'source_best.pth')
            self.assertEqual(best['iteration'], 1)
            with zipfile.ZipFile(args.output_dir/'results_portable.zip') as archive:
                self.assertEqual(len(archive.namelist()), 3)
                self.assertTrue(all(Path(n).suffix in ('.json', '.jsonl') for n in archive.namelist()))
            # Source selection is recorded by the runner; valid scheduled snapshots also load.
            best['training_settings']['steps'] = 20000
            with patch('scripts.snd_remote.load_checkpoint', return_value=best):
                SNDRemoteRuntime.load_source('unused')
                best['checkpoint_selection'] = 'last_scheduled'
                best['iteration'] = 15000
                SNDRemoteRuntime.load_source('unused')
                best['iteration'] = 20001
                with self.assertRaises(ValueError):
                    SNDRemoteRuntime.load_source('unused')
                best['iteration'] = 0
                with self.assertRaises(ValueError):
                    SNDRemoteRuntime.load_source('unused')


if __name__ == '__main__':
    unittest.main()
