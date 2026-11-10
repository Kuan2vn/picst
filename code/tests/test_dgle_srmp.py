"""Check paired geometry, the original ReMask equations, SRMP gradient routing,
and deterministic preemption recovery of the combined continuation runner.
"""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import train_DGLE_SRMP_remote as runner
from train_DGLE_remote import PseudoTraining
from train_SRMP_remote import load_checkpoint, rng_state
from scripts.snd_srmp import RelationalAux
from core.models.segformer import SegFormerB2Decoder


class Images:
    crop = 32
    def __len__(self):
        return 2
    def raw(self, index):
        return Image.fromarray(np.random.RandomState(index).randint(0, 256, (32, 32, 3), dtype=np.uint8))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (3, 64, 128, 320, 512)
        self.layers = nn.ModuleList(nn.Conv2d(channels[i], channels[i+1], 3,
            stride=4 if i == 0 else 2, padding=1) for i in range(4))
    def forward(self, value):
        features = []
        for layer in self.layers:
            value = layer(value); features.append(value)
        return value, features


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.classifier = Encoder(), SegFormerB2Decoder()
    def forward(self, images, size=None):
        return self.classifier(self.encoder(images)[1], size or images.shape[-2:])


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_same_geometry_as_dgle_and_canonical_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid = (np.indices((32, 32)).sum(0) % 6).astype(np.uint8)
            Image.fromarray(grid).save(root / '000000.png')
            original, new = PseudoTraining(Images(), root), runner.PairedTraining(Images(), root)
            for seed in range(6):
                expected_image, expected_label = original[(0, seed)]
                image, label, _, _, geometry, _ = new[(0, seed)]
                torch.testing.assert_close(image, expected_image, rtol=0, atol=0)
                torch.testing.assert_close(label, expected_label, rtol=0, atol=0)
                aligned = runner.align_map(torch.tensor(grid)[None, None], geometry[None], 'nearest')
                torch.testing.assert_close(aligned[0, 0].long(), label, rtol=0, atol=0)

    def test_temporal_and_balance_match_existing_remask(self):
        tree = ast.parse(Path('train_ReMask.py').read_text(encoding='utf-8'))
        names = {'_cfg_get', '_compute_temporal_label_reliability', '_compute_class_balance_reliability'}
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'train_ReMask.py', 'exec'), namespace)
        class Config(SimpleNamespace):
            def __contains__(self, key):
                return hasattr(self, key)
        cfg = SimpleNamespace(REMASK=Config(TEMPORAL_FLOOR=.5, HISTORY_USE_CONF=True,
            TEMPORAL_CONF_POWER=0., EPSILON=1e-6, CLASS_BALANCE_GAMMA=.25,
            CLASS_BALANCE_MIN=.7, CLASS_BALANCE_MAX=1.3), MODEL=SimpleNamespace(NUM_CLASSES=6),
            INPUT=SimpleNamespace(IGNORE_LABEL=255))
        memory = runner.ReMaskMemory()
        geometry = torch.tensor([[8, 0, 0, 8, 0]])
        previous = torch.randn(1, 6, 8, 8).softmax(1)
        memory.update(previous, torch.tensor([2]))
        current = torch.randn(1, 6, 8, 8).softmax(1)
        labels, weight, _ = memory.targets(current.log(), geometry, torch.tensor([2]))
        hist_y, hist_q = memory.history[2]
        temporal = namespace['_compute_temporal_label_reliability'](cfg, current, hist_y[None], hist_q[None])
        balance = namespace['_compute_class_balance_reliability'](cfg, labels, current.mean((0, 2, 3)))
        torch.testing.assert_close(weight[:, None], temporal * balance, rtol=1e-6, atol=1e-7)
        memory.update(current, torch.tensor([2]))
        expected = (.95 * hist_q.float() + .05 * current.max(1, keepdim=True).values[0].half().float()).half()
        torch.testing.assert_close(memory.history[2][1], expected, rtol=0, atol=0)
        restored = runner.ReMaskMemory(); restored.load_state_dict(memory.state_dict(), 'cpu')
        y1, w1, _ = memory.targets(current.log(), geometry, torch.tensor([2]))
        y2, w2, _ = restored.targets(current.log(), geometry, torch.tensor([2]))
        torch.testing.assert_close(y1, y2); torch.testing.assert_close(w1, w2)

    def test_phase1_entropy_kl_matches_original_and_restores_running_stats(self):
        tree = ast.parse(Path('train_ReMask.py').read_text(encoding='utf-8'))
        names = {'_cfg_get', '_update_running_scalar_stats', '_normalize_uncertainty',
                 '_compute_reliability', '_apply_loss_weight_floor'}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'train_ReMask.py', 'exec'), namespace)
        cfg = runner.remask_config(SimpleNamespace(task='vh2pd'))
        memory = runner.ReMaskEntropyKL(cfg)
        expected_stats = {key: None for key in memory.uncertainty}
        torch.manual_seed(34)
        for _ in range(3):
            teacher = torch.randn(2, 6, 8, 8).softmax(1).requires_grad_()
            student = torch.randn(2, 6, 8, 8, requires_grad=True)
            expected = namespace['_compute_reliability'](cfg, teacher.detach(),
                student.detach().softmax(1), expected_stats)
            expected = namespace['_apply_loss_weight_floor'](cfg, expected)[:, 0]
            actual, record = memory.weights(teacher, student)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(actual.requires_grad)
            self.assertEqual(record['remask_phase'], 'entropy_kl')
            for key in expected_stats:
                torch.testing.assert_close(memory.uncertainty[key], expected_stats[key], rtol=0, atol=0)
        ce = torch.nn.functional.cross_entropy(student, teacher.argmax(1), reduction='none')
        loss = (actual * ce).sum() / (actual.sum() + 1e-6)
        loss.backward()
        self.assertIsNone(teacher.grad)
        torch.testing.assert_close(student.grad, (student.detach().softmax(1) -
            torch.nn.functional.one_hot(teacher.argmax(1), 6).permute(0, 3, 1, 2)) *
            actual[:, None] / (actual.sum() + 1e-6), rtol=1e-5, atol=1e-7)
        restored = runner.ReMaskEntropyKL(cfg)
        restored.load_state_dict(memory.state_dict(), 'cpu')
        torch.testing.assert_close(memory.weights(teacher, student)[0],
            restored.weights(teacher, student)[0], rtol=0, atol=0)
        self.assertNotIn('history', memory.state_dict())
        with self.assertRaises(ValueError):
            restored.load_state_dict(runner.ReMaskMemory().state_dict(), 'cpu')

    def test_source_remask_does_not_open_dense_or_training_gt(self):
        dataset = runner.PairedTraining(Images(), None)
        with patch.object(Image, 'open', side_effect=AssertionError('Unexpected target label read')):
            image, label, clean, full, geometry, index = dataset[(0, 22)]
        self.assertTrue((label == 255).all())
        self.assertEqual(image.shape, clean.shape)
        self.assertEqual(full.shape, (3, 32, 32))

    def test_frozen_bn_keeps_bf16_activations_and_exact_fp32_path(self):
        from core.models.layers import FrozenBatchNorm2d
        bn = FrozenBatchNorm2d(3)
        image = torch.randn(2, 3, 4, 8)
        scale = (bn.weight * bn.running_var.rsqrt()).view(1, 3, 1, 1)
        bias = (bn.bias - bn.running_mean * scale.view(3)).view(1, 3, 1, 1)
        torch.testing.assert_close(bn(image), image * scale + bias, rtol=0, atol=0)
        self.assertEqual(bn(image.bfloat16()).dtype, torch.bfloat16)

    def test_suite_runs_four_fresh_adaptations_and_only_one_dgle_round(self):
        bash = shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file():
            self.skipTest('bash unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log, stub = root / 'commands.tsv', root / 'python-stub.sh'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\t" "$@" >> "$SUITE_STUB_LOG"\nprintf "\\n" >> "$SUITE_STUB_LOG"\n', encoding='utf-8')
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON=stub.as_posix(), G2C_REPO=root.as_posix(),
                G2C_SOURCE_CKPT='pretrained/G2C_model_iter020000.pth',
                SUITE_STUB_LOG=log.as_posix(), SEED='1', EXPERIMENT_ID='stub_suite')
            env.pop('PREFLIGHT_ONLY', None)
            subprocess.run([bash, 'run_source_aux_suite.sh'], env=env, check=True, capture_output=True)
            commands = [line.strip().split('\t') for line in log.read_text().splitlines()]
            adaptation = [c for c in commands if c[:2] == ['train_DGLE_SRMP_remote.py', 'adapt'] and '--preflight' not in c]
            self.assertEqual([c[c.index('--method')+1] for c in adaptation],
                ['relation', 'remask', 'remask-relation', 'remask-relation'])
            self.assertTrue(all('--from-source' in c for c in adaptation))
            self.assertNotIn('--remask-phase', adaptation[0])
            self.assertTrue(all(c[c.index('--remask-phase')+1] == 'entropy_kl' for c in adaptation[1:]))
            diffusion = [c for c in commands if c[0] == 'train_DGLE_remote.py']
            self.assertEqual([c[1] for c in diffusion], ['seeds', 'diffusion', 'propagate'])
            self.assertTrue(all(c[c.index('--rounds')+1] == '1' for c in diffusion))
            self.assertEqual(adaptation[-1][adaptation[-1].index('--task')+1], 'g2c')
            self.assertEqual(adaptation[-1][adaptation[-1].index('--updates')+1], '30000')
            self.assertEqual(adaptation[-1][adaptation[-1].index('--schedule-updates')+1], '30000')
            self.assertNotIn('--bf16', adaptation[-1])

    def test_fresh_source_ignores_pretraining_optimizer_and_iteration(self):
        initial_model = Model()
        parent = dict(iteration=19000, seed=999, optimizer='must not load', rng='must not restore')
        source = {'classifier': initial_model.classifier.state_dict()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / 'source.pth'; initial.write_bytes(b'source weights')
            args = SimpleNamespace(seed=1, device='cpu', initial=initial, source=initial,
                from_source=True, method='remask', remask_phase='entropy_kl', aux_weight=0., aux_prepare=0, aux_ramp=1,
                hidden_dim=64, depth=1, refine_lr=2e-5, updates=1, batch_size=2,
                workers=0, labels=None, smoke=True, eval_period=1, schedule_updates=2,
                aux_gate='none', encoder_grad_cap=.25, ema_period=1, ema=.9, output=root)
            snapshots = []
            original = runner.write_checkpoint
            def save(path, state):
                if state['iteration'] == 0:
                    snapshots.append(copy.deepcopy(state))
                original(path, state)
            with patch.object(runner, 'load_segmentation', lambda *v: copy.deepcopy(initial_model)), \
                 patch.object(runner, 'TargetImages', lambda _: Images()), \
                 patch.object(runner, 'evaluate_segmentation', lambda *v: None), \
                 patch.object(runner, 'export_report', lambda _: None), \
                 patch.object(runner, 'write_checkpoint', save):
                runner.train(args, 'fresh-source', parent, source)
            self.assertEqual(snapshots[0]['parent_iteration'], 0)
            self.assertEqual(snapshots[0]['optimizer']['state'], {})
            last = load_checkpoint(root / 'last.pth')
            self.assertEqual(last['absolute_iteration'], 1)
            self.assertEqual(last['memory']['mode'], 'entropy_kl')
            self.assertTrue(all(torch.isfinite(value) for value in last['memory']['uncertainty'].values()))
            self.assertNotIn('history', last['memory'])

    def test_aspp_mrm_gradient_routing_and_anchor_resume(self):
        from scripts.g2c_source_aux import ASPPAux
        from core.models.classifier import ASPP_Classifier_V2
        encoder = nn.Conv2d(3, 2048, 1, stride=2)
        classifier = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], 19)
        source = copy.deepcopy(classifier).requires_grad_(False).eval()
        settings = dict(weight=.1, warmup_updates=0, ramp_updates=1, temperature=1.,
            mask_ratio=.4, hidden_dim=64, depth=1, weight_decay=.01)
        aux = ASPPAux(source, settings, 'cpu', 1, 2e-5)
        features = encoder(torch.randn(1, 3, 8, 16))
        teacher_features = features.detach()
        teacher_logits = source(teacher_features, (8, 16)).detach()
        dense = teacher_logits.argmax(1)
        main = classifier(features, (8, 16)).square().mean()
        main.backward(retain_graph=True)
        head_gradients = [p.grad.clone() for p in classifier.parameters()]
        loss, stats = runner.relation_loss(aux, source, features, teacher_features,
            teacher_logits, dense, 'none')
        self.assertGreater(stats['aux_selected_cells'], 0)
        record = runner.add_aux_gradients(loss, encoder, aux, .1, False, .25)
        self.assertGreater(record['aux_grad_ratio'], 0)
        for p, expected in zip(classifier.parameters(), head_gradients):
            torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in source.parameters()))
        restored = ASPPAux(source, settings, 'cpu', 1, 2e-5)
        restored.load_state_dict(aux.state_dict())
        expected = torch.rand((1, 1, 16, 16), generator=aux.mask_rng)
        actual = torch.rand((1, 1, 16, 16), generator=restored.mask_rng)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cityscapes_rgb_only_geometry_and_native_validation_labels(self):
        from scripts.g2c_source_aux import TrainingImages, ValidationImages, align_city
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_name = 'aachen/aachen_000000_000019_leftImg8bit.png'
            val_name = 'frankfurt/frankfurt_000000_000019_leftImg8bit.png'
            for split, name in (('train', train_name), ('val', val_name)):
                image_path = root / 'data' / split / name
                image_path.parent.mkdir(parents=True)
                pixels = np.random.RandomState(7).randint(0, 256, (12, 24, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(image_path)
            label_path = root / 'label/val' / val_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            label_path.parent.mkdir(parents=True)
            raw = np.full((12, 24), 7, dtype=np.uint8); raw[:2] = 0
            Image.fromarray(raw).save(label_path)
            train_list, val_list = root / 'train.txt', root / 'val.txt'
            train_list.write_text(train_name); val_list.write_text(val_name)
            args = SimpleNamespace(city_root=root, city_train_list=train_list, city_val_list=val_list,
                city_config=Path('configs/deeplabv2_r101_SND_G2C.yaml'), smoke=True, data_limit=0, eval_limit=0)
            dataset = TrainingImages(args)
            batch = next(iter(DataLoader(dataset, batch_sampler=[[(0, 17)]])))
            strong, dummy, clean, full, geometry, index = batch
            self.assertEqual(strong.shape, (1, 3, 512, 1024))
            self.assertTrue((dummy == 255).all())
            torch.testing.assert_close(clean, align_city(full, geometry), rtol=0, atol=0)
            # There is deliberately no label/train directory in this fixture.
            image, labels = ValidationImages(args)[0]
            self.assertEqual(image.shape, (3, 768, 1536))
            self.assertEqual(labels.shape, (12, 24))
            self.assertTrue((labels[:2] == 255).all())
            self.assertTrue((labels[2:] == 0).all())

    def test_aux_reaches_encoder_and_rebuilder_but_not_either_classifier(self):
        torch.manual_seed(7)
        model = Model().train()
        teacher = copy.deepcopy(model).requires_grad_(False).eval()
        source = copy.deepcopy(model.classifier).requires_grad_(False).eval()
        settings = dict(rebuilder='mrm', objective='relation', weight=.1, warmup_updates=0,
            ramp_updates=1, mask_ratio=.4, block_size=1, temperature=1., hidden_dim=64, depth=1, weight_decay=.01)
        aux = RelationalAux({'classifier': source.state_dict()}, settings, 'cpu', 1, 2e-5)
        images = torch.randn(2, 3, 32, 32)
        features = model.encoder(images)[1]
        with torch.no_grad():
            teacher_features = teacher.encoder(images)[1]
            teacher_logits = teacher.classifier(teacher_features, (32, 32))
        dense = teacher_logits.argmax(1)
        main = model.classifier(features, (32, 32)).square().mean()
        main.backward(retain_graph=True)
        head_gradients = [p.grad.clone() for p in model.classifier.parameters()]
        loss, stats = runner.relation_loss(aux, source, features, teacher_features, teacher_logits, dense, 'none')
        self.assertGreater(stats['aux_selected_cells'], 0)
        record = runner.add_aux_gradients(loss, model.encoder, aux, .1, False, .25)
        self.assertGreater(record['aux_grad_ratio'], 0)
        for p, expected in zip(model.classifier.parameters(), head_gradients):
            torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in source.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in aux.rebuilder.parameters()))
        # Explicitly empty reliability gate stays finite and adds no bogus targets.
        loss, stats = runner.relation_loss(aux, source, [f.detach() for f in features],
            teacher_features, teacher_logits, torch.full_like(dense, 255), 'agreement')
        self.assertEqual(stats['aux_selected_cells'], 0); self.assertEqual(loss.item(), 0.)

    def test_combined_resume_restores_teacher_memory_optimizer_and_aux_rng(self):
        torch.manual_seed(19)
        initial_model = Model()
        optimizer = torch.optim.SGD([{'params': initial_model.encoder.parameters()},
            {'params': initial_model.classifier.parameters()}], lr=2e-5, momentum=.9, weight_decay=.01)
        parent = dict(iteration=2, seed=1, optimizer=optimizer.state_dict(), rng=rng_state())
        source = {'classifier': initial_model.classifier.state_dict()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / 'initial.pth'; initial.write_bytes(b'parent checkpoint')
            labels = root / 'dense'; labels.mkdir()
            for index in range(2):
                grid = (np.indices((32, 32)).sum(0) % 6).astype(np.uint8)
                Image.fromarray(grid).save(labels / f'{index:06d}.png')
            args = SimpleNamespace(seed=1, device='cpu', initial=initial, source=initial, method='remask-relation',
                remask_phase='entropy_kl',
                aux_weight=.1, aux_prepare=1, aux_ramp=1, hidden_dim=64, depth=1,
                refine_lr=2e-5, updates=3, batch_size=2, workers=0, labels=labels,
                smoke=True, eval_period=1, schedule_updates=8, aux_gate='none',
                encoder_grad_cap=.25, ema_period=1, ema=.9)
            def evaluation(args, *values):
                (args.output / 'evaluation_curve.jsonl').touch(exist_ok=True)
            with patch.object(runner, 'load_segmentation', lambda *v: copy.deepcopy(initial_model)), \
                 patch.object(runner, 'TargetImages', lambda _: Images()), \
                 patch.object(runner, 'evaluate_segmentation', evaluation), \
                 patch.object(runner, 'export_report', lambda _: None):
                args.output = root / 'full'; args.output.mkdir()
                runner.train(args, 'test', parent, source)
                args.output = root / 'interrupted'; args.output.mkdir()
                original = runner.write_checkpoint
                def interrupt(path, state):
                    original(path, state)
                    if state['iteration'] == 1:
                        raise InterruptedError('preemption after checkpoint')
                with patch.object(runner, 'write_checkpoint', interrupt):
                    with self.assertRaises(InterruptedError):
                        runner.train(args, 'test', parent, source)
                runner.train(args, 'test', parent, source)
            complete = load_checkpoint(root / 'full/last.pth')
            resumed = load_checkpoint(root / 'interrupted/last.pth')
            def equal(first, second):
                if torch.is_tensor(first):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                elif isinstance(first, dict):
                    self.assertEqual(first.keys(), second.keys())
                    for key in first:
                        equal(first[key], second[key])
                elif isinstance(first, (tuple, list)):
                    for a, b in zip(first, second):
                        equal(a, b)
                else:
                    self.assertEqual(first, second)
            for name in ('feature_extractor', 'classifier', 'teacher', 'optimizer', 'memory', 'auxiliary'):
                equal(complete[name], resumed[name])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
