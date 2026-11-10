"""CPU checks for protocol boundaries and native-mask/stride correctness."""
import argparse
import copy
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from scripts.prepare_siamseg_data import COLORS, native_mask, positions
from scripts.siamseg_reproduce import build_config, report


class SiamSegReproductionTest(unittest.TestCase):
    def test_native_labels_are_reduced_once(self):
        rgb = Image.fromarray(np.array([[(0, 0, 0), *COLORS]], dtype=np.uint8))
        native = np.asarray(native_mask(rgb))
        np.testing.assert_array_equal(native, [[0, 1, 2, 3, 4, 5, 6]])
        reduced = np.where(native == 0, 255, native.astype(int) - 1)
        np.testing.assert_array_equal(reduced, [[255, 0, 1, 2, 3, 4, 5]])

    def test_stride_does_not_overwrite_distinct_patches(self):
        self.assertEqual(positions(1024, 512, 256), [0, 256, 512])
        self.assertEqual(positions(1100, 512, 256), [0, 256, 512, 588])
        self.assertEqual(len(positions(6000, 512, 512)), 12)

    def test_source_has_optimizer_hook_and_uda_starts_independently(self):
        original = {
            'work_dir': 'original', 'model': {'train_cfg': {'work_dir': 'original'},
                'pretrained': 'old', 'decode_head': {'in_channels': [64, 128, 320, 512],
                  'in_index': [0, 1, 2, 3], 'dropout_ratio': .1, 'num_classes': 7,
                  'norm_cfg': {'type': 'SyncBN'}, 'loss_decode': {'type': 'CrossEntropyLoss'}}},
            'data': {'train': {'type': 'UDADataset', 'source': {'pipeline': [('native',)]},
                              'target': {'pipeline': [('target-native',)]}}, 'val': {}, 'test': {}},
            'uda': {'type': 'SiamSeg', 'alpha': .999, 'contras_model_cfg': {'mod': 'target'}},
            'optimizer_config': None, 'load_from': 'source_weights', 'resume_from': 'adapt_weights',
        }
        options = dict(task='r2u', data_root=Path('native'), backbone=Path('pretrain'), work_dir=Path('run'))
        source = build_config(original, stage='source', **options)
        self.assertNotIn('uda', source)
        self.assertEqual(source['optimizer_config'], {})
        self.assertNotIn('target', source['data']['train'])
        uda = build_config(original, stage='adapt', **options)
        self.assertIsNone(uda['load_from'])
        self.assertIsNone(uda['optimizer_config'])
        self.assertEqual(uda['uda'], original['uda'])
        self.assertEqual(uda['data']['train']['target']['pipeline'], [('target-native',)])
        self.assertEqual(original['load_from'], 'source_weights')

    def test_portable_metrics_from_upstream_pre_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            protocol = {'backbone_sha256': 'pretrain', 'preprocess': {}, 'paper_protocol_verified': False}
            (work / 'protocol_reproduce.json').write_text(json.dumps(protocol))
            (work / 'generated_config.py').write_text('data = {}\n')
            checkpoint = work / 'iter_40000.pth'
            checkpoint.write_bytes(b'never deserialized into report')
            arrays = (np.ones(7), np.full(7, 2), np.ones(7), np.full(7, 2))
            with (work / 'eval_pre_eval.pkl').open('wb') as stream:
                pickle.dump([arrays, arrays], stream)
            report(argparse.Namespace(work_dir=work, checkpoint=checkpoint, task='r2u', stage='adapt'))
            result = json.loads((work / 'result_portable.json').read_text())
            self.assertFalse(result['contains_weights'])
            self.assertEqual(result['metrics']['miou'], .5)
            self.assertEqual(result['metrics']['valid_pixels'], 28)
            self.assertEqual(result['metrics']['evaluation_images'], 2)
            self.assertEqual(result['published_reference']['miou'], .5172)

    def test_real_upstream_config_and_tuple_roundtrip(self):
        # MMCV 1.x is required for the native training environment. Run this
        # test with that environment; skip only when it is absent locally.
        try:
            from mmcv import Config
        except ImportError:
            self.skipTest('MMCV 1.x unavailable')
        from scripts.siamseg_reproduce import CONFIGS, ROOT, verify_upstream
        import pprint
        repo = ROOT / 'baselinescode/SiamSeg'
        verify_upstream(repo)
        for task in ('r2u', 'vh2pd'):
            original = Config.fromfile(str(repo / 'configs/SiamSeg' / CONFIGS[task]))._cfg_dict.to_dict()
            for stage in ('source', 'source-segformer', 'dacs', 'adapt'):
                cfg = build_config(original, task=task, stage=stage, data_root=Path('native'),
                                   backbone=Path('pretrain'), work_dir=Path('run'))
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / 'generated_config.py'
                    path.write_text('\n'.join(f'{k} = {pprint.pformat(v)}' for k, v in cfg.items()))
                    loaded = Config.fromfile(str(path))
                    train = loaded.data.train.source if stage in ('dacs', 'adapt') else loaded.data.train
                    resize = next(step for step in train.pipeline if step['type'] == 'Resize')
                    self.assertIsInstance(resize.img_scale, tuple)
                    self.assertEqual(loaded.runner.max_iters, 40000)
                    self.assertEqual(loaded.optimizer.lr, .00006)
                    self.assertIsNone(loaded.load_from)
                    self.assertEqual(loaded.model.decode_head.num_classes, 7 if task == 'r2u' else 6)


if __name__ == '__main__':
    unittest.main()
