"""Check DGLE seed semantics, DDIM, DDP gradients and deterministic resume."""
import tempfile
import unittest
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.dgle import (DDPSegmenter, DeformableAttention, alpha_sigma, ddim_update,
                          fuse_seeds, percentile_index, sparse_ce, RRDBNet, enhance)
from train_DGLE_remote import PlannedBatches, global_thresholds, SR_SHA
import train_DGLE_remote as trainer
import train_DGLE_remote as trainer
from train_SRMP_remote import load_checkpoint, sha256_file


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(3 if i == 0 else channels[i - 1], channels[i], 3,
                                            stride=4 if i == 0 else 2, padding=1)
                                    for i in range(4) for channels in [(64, 128, 320, 512)])

    def forward(self, x):
        result = []
        for layer in self.layers:
            x = layer(x); result.append(x)
        return x, result


class DGLEChecks(unittest.TestCase):
    def test_exact_global_percentile_and_intersection(self):
        label = np.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=np.uint8)
        score = np.array([[[.1, .9, .2, .8]], [[.3, .7, .4, .6]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            thresholds, counts = global_thresholds(label, score, .6, Path(directory))
        np.testing.assert_allclose(thresholds[:2], [.7, .6])
        self.assertEqual(counts, [4, 4, 0, 0, 0, 0])
        sr = label.copy(); sr[0, 0, 1] = 1  # High confidence but disagrees: must be ignored.
        seeds = fuse_seeds(label, score, sr, score, thresholds, thresholds)
        np.testing.assert_array_equal(seeds, [[[255, 255, 255, 1]], [[255, 0, 255, 1]]])
        self.assertEqual(percentile_index(10, .6), 5)
        self.assertEqual(percentile_index(1, 1), 0)

    def test_ddim_recovers_consistent_noisy_state(self):
        clean, noise = torch.randn(2, 8, 3, 4), torch.randn(2, 8, 3, 4)
        now, nxt = torch.tensor([.8, .5]), torch.tensor([.4, 0.])
        a, s = alpha_sigma(now); an, sn = alpha_sigma(nxt)
        noisy = a[:, None, None, None] * clean + s[:, None, None, None] * noise
        expected = an[:, None, None, None] * clean + sn[:, None, None, None] * noise
        torch.testing.assert_close(ddim_update(noisy, clean, now, nxt), expected)

    def test_three_step_sampler_matches_upstream_when_no_seeds(self):
        # Execute the pinned author's sampler, with only its framework adapters
        # replaced. This checks actual timestep lookahead, argmax re-embedding,
        # DDIM update, and final-logit selection against the original code.
        from einops import repeat
        path = Path('references/dgle/ddp_upstream.py')
        tree = ast.parse(path.read_text())
        names = {'log', 'alpha_cosine_log_snr', 'log_snr_to_alpha_sigma'}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DDP')
        selected = {'right_pad_dims_to', '_get_sampling_timesteps', 'ddim_sample'}
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        reference_class = ast.ClassDef(name='Reference', bases=[ast.Name(id='object', ctx=ast.Load())],
                                      keywords=[], body=methods, decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=nodes + [reference_class], type_ignores=[]))
        namespace = dict(torch=torch, math=math, repeat=repeat)
        exec(compile(module, str(path), 'exec'), namespace)
        model = DDPSegmenter(TinyEncoder(), channels=32, layers=2).eval()
        images = torch.randn(1, 3, 64, 64)
        seeds = torch.full((1, 64, 64), 255, dtype=torch.long)
        reference = namespace['Reference']()
        reference.timesteps, reference.sample_range = 3, (0, .999)
        reference.time_difference, reference.randsteps, reference.accumulation = 1, 1, False
        reference.decode_head = SimpleNamespace(in_channels=[32])
        reference.embedding_table, reference.bit_scale = model.embedding, model.bit_scale
        reference.transform, reference.time_mlp = model.transform, model.time_mlp
        reference.log_snr = namespace['alpha_cosine_log_snr']
        def decode(features, times, img_metas=None):
            x = features[0]; batch, channels, height, width = x.shape
            from scripts.dgle import sine_position
            position = sine_position(height, width, channels, x.device)
            query = x.flatten(2).transpose(1, 2)
            for layer in model.layers:
                query = layer(query, position, times, height, width)
            return model.readout(query.transpose(1, 2).reshape(batch, channels, height, width))
        reference._decode_head_forward_test = decode
        with torch.no_grad():
            condition = model.condition(images)
            torch.manual_seed(21)
            author_logits = reference.ddim_sample(condition, None)
            torch.manual_seed(21)
            actual = model.sample(images, seeds, steps=3)
            expected = torch.nn.functional.interpolate(author_logits, size=(64, 64), mode='bilinear', align_corners=False)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_attention_center_sampling_equivalence(self):
        # One sample at each pixel center must reduce to value projection + residual.
        attention = DeformableAttention(32, heads=8, points=1)
        with torch.no_grad():
            attention.sampling_offsets.bias.zero_()
        query = torch.randn(2, 15, 32, requires_grad=True)
        expected = query + attention.output_proj(attention.value_proj(query))
        actual = attention(query, torch.zeros(1, 15, 32), 3, 5)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_sparse_ddp_and_seeded_three_step_sampling(self):
        torch.manual_seed(8)
        model = DDPSegmenter(TinyEncoder(), channels=32, layers=2)
        images = torch.randn(2, 3, 32, 32)
        seeds = torch.full((2, 32, 32), 255, dtype=torch.long)
        seeds[:, :16, :16] = 1
        loss, _ = model.training_loss(images, seeds)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for module in (model.encoder, model.embedding, model.time_mlp, model.layers):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        zero = sparse_ce(torch.randn(1, 6, 2, 2, requires_grad=True), torch.full((1, 2, 2), 255))
        self.assertEqual(zero.item(), 0); zero.backward()
        model.eval()
        captured, original = [], model.denoise
        def observe(condition, state, times):
            captured.append(state.clone())
            return original(condition, state, times)
        model.denoise = observe
        noise = torch.randn(2, 32, 8, 8)
        prediction = model.sample(images, seeds, steps=3, noise=noise)
        embedded = model.encode_labels(seeds, (8, 8))
        torch.testing.assert_close(captured[0][:, :, :4, :4], embedded[:, :, :4, :4])
        torch.testing.assert_close(captured[0][:, :, 4:, :], noise[:, :, 4:, :])
        self.assertEqual(len(captured), 3)
        self.assertEqual(prediction.shape, (2, 6, 32, 32))

    def test_resume_batches_identical(self):
        batches = list(PlannedBatches(7, 3, 10, 0, 17))
        resumed = list(PlannedBatches(7, 3, 10, 4, 17))
        self.assertEqual(batches[4:], resumed)
        self.assertEqual(len({i for batch in batches[:3] for i, _ in batch}), 7)

    def test_interrupted_diffusion_resume_matches_uninterrupted_weights(self):
        from PIL import Image
        class Images:
            crop = 32
            def __len__(self):
                return 2
            def raw(self, index):
                return Image.fromarray(np.random.RandomState(index).randint(0, 256, (32, 32, 3), dtype=np.uint8))
        args = SimpleNamespace(device='cpu', seed=9, amp=False, workers=0, batch_size=2,
                               diffusion_updates=3, diffusion_epochs=20, diffusion_lr=6e-5,
                               eval_period=1, checkpoint_period=1, smoke=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); initial = root / 'initial'; initial.write_bytes(b'smoke initializer')
            for name in ('full', 'interrupted'):
                directory = root / name / 'seeds'; directory.mkdir(parents=True)
                for index in range(2):
                    seed = np.full((32, 32), 255, dtype=np.uint8); seed[:16] = index
                    Image.fromarray(seed).save(directory / f'{index:06d}.png')
            build = lambda _: DDPSegmenter(TinyEncoder(), channels=32, layers=2)
            with patch.object(trainer, 'build_diffusion', build):
                trainer.seed_all(9)
                trainer.train_stage(args, Images(), root / 'full', initial, 'test', 1, 'diffusion')
                save = trainer.save_training
                def interrupt_after_save(*values, **kwargs):
                    save(*values, **kwargs)
                    raise InterruptedError('Simulated preemption after durable checkpoint')
                trainer.seed_all(9)
                with patch.object(trainer, 'save_training', interrupt_after_save):
                    with self.assertRaises(InterruptedError):
                        trainer.train_stage(args, Images(), root / 'interrupted', initial, 'test', 1, 'diffusion')
                trainer.seed_all(100)  # Resume must restore its checkpoint RNG.
                trainer.train_stage(args, Images(), root / 'interrupted', initial, 'test', 1, 'diffusion')
            full = load_checkpoint(root / 'full/diffusion/last.pth')
            resumed = load_checkpoint(root / 'interrupted/diffusion/last.pth')
            for name, weight in full['diffusion'].items():
                torch.testing.assert_close(weight, resumed['diffusion'][name], rtol=0, atol=0)

    def test_interrupted_diffusion_resume_matches_uninterrupted_weights(self):
        from PIL import Image
        class Images:
            crop = 32
            def __len__(self):
                return 2
            def raw(self, index):
                return Image.fromarray(np.random.RandomState(index).randint(0, 256, (32, 32, 3), dtype=np.uint8))
        args = SimpleNamespace(device='cpu', seed=9, amp=False, workers=0, batch_size=2,
                               diffusion_updates=3, diffusion_epochs=20, diffusion_lr=6e-5,
                               eval_period=1, checkpoint_period=1, smoke=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); initial = root / 'initial'; initial.write_bytes(b'smoke initializer')
            for name in ('full', 'interrupted'):
                directory = root / name / 'seeds'; directory.mkdir(parents=True)
                for index in range(2):
                    seed = np.full((32, 32), 255, dtype=np.uint8); seed[:16] = index
                    Image.fromarray(seed).save(directory / f'{index:06d}.png')
            build = lambda _: DDPSegmenter(TinyEncoder(), channels=32, layers=2)
            with patch.object(trainer, 'build_diffusion', build):
                trainer.seed_all(9)
                trainer.train_stage(args, Images(), root / 'full', initial, 'test', 1, 'diffusion')
                save = trainer.save_training
                def interrupt_after_save(*values, **kwargs):
                    save(*values, **kwargs)
                    raise InterruptedError('Simulated preemption after durable checkpoint')
                trainer.seed_all(9)
                with patch.object(trainer, 'save_training', interrupt_after_save):
                    with self.assertRaises(InterruptedError):
                        trainer.train_stage(args, Images(), root / 'interrupted', initial, 'test', 1, 'diffusion')
                trainer.seed_all(100)  # Resume must restore its checkpoint RNG.
                trainer.train_stage(args, Images(), root / 'interrupted', initial, 'test', 1, 'diffusion')
            full = load_checkpoint(root / 'full/diffusion/last.pth')
            resumed = load_checkpoint(root / 'interrupted/diffusion/last.pth')
            for name, weight in full['diffusion'].items():
                torch.testing.assert_close(weight, resumed['diffusion'][name], rtol=0, atol=0)

    def test_official_sr_checkpoint(self):
        path = Path('pretrained/dgle/RealESRGAN_x4plus.pth')
        if not path.exists():
            self.skipTest('Real-ESRGAN weights not present')
        self.assertEqual(sha256_file(path), SR_SHA)
        model = RRDBNet().eval()
        model.load_state_dict(load_checkpoint(path)['params_ema'], strict=True)
        image = np.random.RandomState(3).randint(0, 256, (8, 8, 3), dtype=np.uint8)
        result = enhance(model, image, 'cpu', tile=8)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(np.array_equal(result, image))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
