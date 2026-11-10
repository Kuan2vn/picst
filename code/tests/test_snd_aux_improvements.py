"""Runnable regression checks for R preparation and soft relational discrimination."""
import copy
import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F
from core.models.segformer import build_segformer_b2
from scripts.snd_srmp import RelationalAux, relational_with_soft_kl
from train_SRMP import masked_mean, relational_cosine_map


def test_soft_kl():
    gram = torch.tensor([[1., .1, 0.], [.1, 1., -.2], [0., -.2, 1.]])
    target = torch.tensor([-.0, 8., 0.]).reshape(1, 3, 1, 1).softmax(1).requires_grad_()
    wrong = torch.tensor([20., 0., 0.]).reshape(1, 3, 1, 1).requires_grad_()
    old = relational_cosine_map(wrong.softmax(1), target, gram)
    disabled, _, _ = relational_with_soft_kl(wrong, target, gram, 0)
    assert torch.equal(old, disabled)
    hybrid, relation, kl = relational_with_soft_kl(wrong, target, gram, .1)
    assert torch.equal(old, relation) and torch.allclose(hybrid, relation+.1*kl)
    go = torch.autograd.grad(old.sum(), wrong, retain_graph=True)[0]
    gn = torch.autograd.grad(hybrid.sum(), wrong, retain_graph=True)[0]
    assert go.norm() < 1e-6 and gn.norm() > .1
    assert gn[0, 0] > 0 and gn[0, 1] < 0
    hybrid.sum().backward()
    assert target.grad is None and torch.isfinite(wrong.grad).all()
    same = target.detach().log().requires_grad_()
    value, _, _ = relational_with_soft_kl(same, target, gram, .1)
    assert value.abs().max() < 1e-6
    zero = masked_mean(hybrid, torch.zeros_like(hybrid, dtype=torch.bool))
    assert zero == 0
    print('Saturated wrong prediction gradients:', float(go.norm()), '->', float(gn.norm()))


def geometry(size):
    return dict(rescale_size=([size], [size]), padding=([0], [0], [0], [0]),
        random_crop_axis=([0], [size], [0], [size]), RandomHorizontalFlip=[False])


def test_preparation_and_resume():
    torch.manual_seed(702)
    student, decoder = build_segformer_b2()
    decoder.eval()
    teacher, td = copy.deepcopy(student).eval(), copy.deepcopy(decoder).eval()
    source = dict(classifier=td.state_dict())
    settings = dict(objective='relation', rebuilder='mrm', weight=.2, warmup_updates=1,
        ramp_updates=1, prepare_updates=1, soft_kl_weight=.1, temperature=1.,
        mask_ratio=.4, block_size=1, hidden_dim=512, depth=2, weight_decay=.01)
    aux = RelationalAux(source, settings, 'cpu', 1, 2e-5)
    image = torch.randn(1, 3, 64, 64)
    main = decoder(student(image)[1]).square().mean(); main.backward()
    params = tuple(student.parameters())+tuple(decoder.parameters())
    before_grads = [None if p.grad is None else p.grad.clone() for p in params]
    before_weights = [p.detach().clone() for p in params]
    r_before = [p.detach().clone() for p in aux.rebuilder.parameters()]
    rng = torch.get_rng_state().clone()
    aux.backward(1, 2e-5, main, student, decoder, teacher, td, image, image, geometry(64))
    assert torch.equal(rng, torch.get_rng_state())
    assert aux.last_record['aux_preparing'] and aux.last_record['aux_weight'] == 0
    assert aux.last_record['weighted_aux_loss'] == 0 and aux.last_record['aux_grad_ratio'] == 0
    for p, g in zip(params, before_grads):
        assert p.grad is None if g is None else torch.equal(p.grad, g)
    aux.step()
    assert any(not torch.equal(p, b) for p,b in zip(aux.rebuilder.parameters(), r_before))
    assert all(torch.equal(p, b) for p,b in zip(params, before_weights))
    assert all(p.grad is None for p in tuple(teacher.parameters())+tuple(td.parameters()))
    saved = copy.deepcopy(aux.state_dict())
    restored = RelationalAux(source, settings, 'cpu', 1, 2e-5); restored.load_state_dict(saved)
    outcomes = []
    for module in (aux, restored):
        student.zero_grad(); decoder.zero_grad()
        torch.manual_seed(100)
        main = decoder(student(image)[1]).square().mean(); main.backward()
        old_grad = student.backbone.patch_embed1.proj.weight.grad.clone()
        module.backward(2, 2e-5, main, student, decoder, teacher, td, image, image, geometry(64))
        assert not module.last_record['aux_preparing'] and module.last_record['aux_weight'] == .2
        assert not torch.equal(student.backbone.patch_embed1.proj.weight.grad, old_grad)
        outcomes.append((copy.deepcopy(module.last_record), [p.grad.clone() for p in module.rebuilder.parameters() if p.grad is not None]))
    assert outcomes[0][0] == outcomes[1][0]
    assert all(torch.equal(a,b) for a,b in zip(outcomes[0][1],outcomes[1][1]))
    inactive = RelationalAux(source, dict(settings, weight=0), 'cpu', 1, 2e-5)
    rng = torch.get_rng_state().clone()
    inactive.backward(1, 2e-5, torch.tensor(1.), None, None, None, None, None, None, None)
    inactive.step()
    assert inactive.forward_count == 0 and not inactive.optimizer.state
    assert torch.equal(rng, torch.get_rng_state())
    print('Actual MRM: preparation updates only R; joint phase gradients and exact resume passed')


def test_old_gated_parity():
    path = Path(__file__).resolve().parents[1]/'reports/aux_improvements_20261002/code_before/snd_srmp.py'
    spec = importlib.util.spec_from_file_location('before_improvements', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    from tests.test_snd_srmp_b2 import Encoder, Decoder
    torch.manual_seed(12)
    initial, head = Encoder(), Decoder()
    teacher, td = copy.deepcopy(initial).eval(), copy.deepcopy(head).eval()
    settings = dict(objective='relation', weight=.6, warmup_updates=0, ramp_updates=1,
        temperature=1., mask_ratio=.4, block_size=1, hidden_dim=16, depth=1,
        weight_decay=.01, view_consistency=True, confidence_threshold=.2)
    image = torch.randn(1, 3, 16, 16); results = []
    for klass in (old.RelationalAux, RelationalAux):
        student, decoder = copy.deepcopy(initial), copy.deepcopy(head)
        module = klass(dict(classifier=td.state_dict()), settings, 'cpu', 1, 2e-5)
        main = decoder.fuse_features(student(image)[1]).square().mean(); main.backward()
        rng = torch.get_rng_state().clone()
        module.backward(20, 2e-5, main, student, decoder, teacher, td, image, image, geometry(16))
        assert torch.equal(rng, torch.get_rng_state())
        results.append((module.last_record, [p.grad for p in tuple(student.parameters())+tuple(decoder.parameters())+tuple(module.rebuilder.parameters())], module.state_dict()))
    assert results[0][0] == results[1][0]
    assert all((a is None and b is None) or torch.equal(a,b) for a,b in zip(results[0][1],results[1][1]))
    assert torch.equal(results[0][2]['mask_rng'], results[1][2]['mask_rng'])
    print('Old gated implementation: exact losses, gradients, logs, RNG when new features off')


if __name__ == '__main__':
    torch.set_num_threads(4)
    test_soft_kl()
    test_preparation_and_resume()
    test_old_gated_parity()
    print('Aux improvements OK')
