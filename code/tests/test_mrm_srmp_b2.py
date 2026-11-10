"""MRM upstream parity, masked information removal and frozen-head bilevel check."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F
import higher

from core.models.mrm_rebuilder import Rebuilder
from core.models.segformer import build_segformer_b2
from scripts.snd_srmp import RelationalAux
from train_SND import freeze_b2_source_readout


def test_upstream_parity():
    upstream = Path(__file__).resolve().parents[1]/'reports/mrm_alignment_20261001/official_rebuilder.py'
    # Upstream release depends on mmseg only for a thin interpolation wrapper.
    old_modules = {name: sys.modules.get(name) for name in ('mmseg', 'mmseg.ops')}
    old_endpoint = os.environ.get('HF_ENDPOINT')
    sys.modules['mmseg'] = types.ModuleType('mmseg')
    ops = types.ModuleType('mmseg.ops'); ops.resize = F.interpolate
    sys.modules['mmseg.ops'] = ops
    try:
        spec = importlib.util.spec_from_file_location('official_mrm_reference', upstream)
        reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    finally:
        for name, value in old_modules.items():
            if value is None: sys.modules.pop(name, None)
            else: sys.modules[name] = value
        if old_endpoint is None: os.environ.pop('HF_ENDPOINT', None)
        else: os.environ['HF_ENDPOINT'] = old_endpoint
    cfg = dict(type='DAFormerHead', in_channels=[64, 128, 320, 512], in_index=[0, 1, 2, 3])
    official, port = reference.Rebuilder(cfg).eval(), Rebuilder(cfg).eval()
    port.load_state_dict(official.state_dict(), strict=True)
    features = [torch.randn(1, c, s, s) for c, s in zip(cfg['in_channels'], [128, 64, 32, 16])]
    with torch.no_grad():
        torch.manual_seed(23); expected = official(features)
        torch.manual_seed(23); actual = port(features)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    hidden = torch.rand(1, 1, 16, 16) < .4
    changed = [torch.where(F.interpolate(hidden.float(), size=x.shape[-2:], mode='nearest').bool(), x+100, x)
               for x in features]
    with torch.no_grad():
        a, b = port(features, hidden), port(changed, hidden)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    for original, rebuilt in zip(features, a):
        visible = ~F.interpolate(hidden.float(), size=original.shape[-2:], mode='nearest').bool().expand_as(original)
        assert torch.equal(original[visible], rebuilt[visible])
    print('Official512 multiscale forward: exact equality; visible preserved; hidden perturb invariant')


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Conv2d(4, 3, 1)
        self.linear_pred = nn.Conv2d(3, 2, 1)

    def forward(self, x):
        return self.linear_pred(self.fusion(x))


def test_frozen_meta_gradients():
    torch.manual_seed(8)
    head = TinyHead()
    source = {k: v.clone() for k, v in head.state_dict().items() if k.startswith('linear_pred.')}
    freeze_b2_source_readout(head, source)
    optimizer = torch.optim.SGD((p for p in head.parameters() if p.requires_grad), lr=.1)
    image, validation = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)
    weights = torch.ones(1, 4, 4, requires_grad=True)
    labels = torch.randint(2, (1, 4, 4))
    with higher.innerloop_ctx(head, optimizer, copy_initial_weights=False) as (virtual, diffopt):
        diffopt.step((F.cross_entropy(virtual(image), labels, reduction='none') * weights).mean())
        assert all(torch.equal(virtual.state_dict()[k], v) for k, v in source.items())
        gradient = torch.autograd.grad(F.cross_entropy(virtual(validation), 1-labels), weights)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(torch.equal(head.state_dict()[k], v) for k, v in source.items())
    with torch.no_grad(): head.linear_pred.weight.add_(.01)
    try: freeze_b2_source_readout(head, source)
    except ValueError: pass
    else: raise AssertionError('An adapted warm readout was accepted as source-frozen')
    print('Frozen meta head unchanged; meta pixel-weight gradients nonzero')


def test_mrm_objective_gradients_and_resume():
    torch.manual_seed(12)
    student, decoder = build_segformer_b2()
    decoder.eval()  # Tiny CPU crop; source decoder BN remains fixed in real training.
    teacher, teacher_decoder = copy.deepcopy(student).eval(), copy.deepcopy(decoder).eval()
    source = dict(classifier={k: v.clone() for k, v in decoder.state_dict().items()})
    image = torch.randn(1, 3, 64, 64)
    geometry = dict(rescale_size=(torch.tensor([64]), torch.tensor([64])),
                    padding=tuple(torch.tensor([0]) for _ in range(4)),
                    random_crop_axis=tuple(torch.tensor([v]) for v in (0, 64, 0, 64)),
                    RandomHorizontalFlip=torch.tensor([False]))
    settings = dict(rebuilder='mrm', hidden_dim=512, depth=2, weight_decay=.01,
        warmup_updates=0, ramp_updates=1, weight=.6, temperature=1., mask_ratio=.4, block_size=1)
    for objective, view_gate in (('relation', False), ('masked-ce', False), ('mrm-ce', False), ('relation', True)):
        student.zero_grad(); decoder.zero_grad()
        resolved = dict(settings, objective=objective, view_consistency=view_gate)
        aux = RelationalAux(source, resolved, 'cpu', 1, 2e-5)
        main = decoder(student(image)[1]).square().mean()
        main.backward()
        head_grad = decoder.linear_pred.weight.grad.clone()
        aux.backward(1, 2e-5, main, student, decoder, teacher, teacher_decoder,
                     image, image, geometry)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in aux.rebuilder.parameters())
        assert student.backbone.patch_embed1.proj.weight.grad.abs().sum() > 0
        assert decoder.linear_fuse[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for p in teacher_decoder.parameters())
        assert not aux.weight.requires_grad and not aux.gram.requires_grad
        if objective == 'mrm-ce':
            assert not torch.equal(head_grad, decoder.linear_pred.weight.grad)
            assert aux.last_record['aux_selected_cells'] == 64*64
        else:
            assert torch.equal(head_grad, decoder.linear_pred.weight.grad)
            assert 0 < aux.last_record['aux_selected_cells'] < 16*16
        aux.step()
        restored = RelationalAux(source, resolved, 'cpu', 1, 2e-5)
        restored.load_state_dict(aux.state_dict())
        assert len(restored.optimizer.state) == len(aux.optimizer.state) > 0
        assert torch.equal(restored.mask_rng.get_state(), aux.mask_rng.get_state())
        assert all(torch.equal(a, b) for a, b in zip(aux.rebuilder.parameters(), restored.rebuilder.parameters()))
        # Disabled auxiliary must preserve global RNG and run no student/teacher forward.
        inactive = RelationalAux(source, dict(resolved, weight=0), 'cpu', 1, 2e-5)
        before = torch.get_rng_state().clone()
        inactive.backward(1, 2e-5, torch.tensor(1.), None, None, None, None, None, None, None)
        assert inactive.forward_count == 0 and torch.equal(before, torch.get_rng_state())
        print(objective, 'view_gate', view_gate, 'loss', aux.last_record['aux_loss'], 'selected', aux.last_record['aux_selected_cells'])


if __name__ == '__main__':
    torch.set_num_threads(4)
    test_upstream_parity()
    test_frozen_meta_gradients()
    test_mrm_objective_gradients_and_resume()
    print('MRM-SRMP upstream, gradient, freeze, meta and resume OK')
