"""Runnable checks for source metric, empty context, routing/cap and exact resume."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from core.models.segformer import build_segformer_b2
from scripts.snd_srmp import RelationalAux, context_relational_loss, source_context_kernel
from tests.test_snd_aux_improvements import geometry


def test_kernel_and_loss():
    torch.manual_seed(31)
    u = F.normalize(torch.randn(6, 12), dim=1).requires_grad_()
    a = (u.detach() @ u.detach().T).requires_grad_()
    q, keys = torch.randn(4, 12), torch.randn(5, 12)
    metric = torch.eye(12) + u.detach().T @ a.detach() @ u.detach()
    assert torch.linalg.eigvalsh(metric).min() >= .99999
    expected = (q @ metric @ keys.T) / ((q @ metric*q).sum(1).sqrt()[:, None]*(keys @ metric*keys).sum(1).sqrt()[None, :])
    assert torch.allclose(source_context_kernel(q, keys, u, a), expected, atol=1e-6)
    assert torch.allclose(source_context_kernel(q, keys, u, a, 0), F.normalize(q) @ F.normalize(keys).T, atol=1e-6)
    z = torch.randn(2, 12, 4, 4, requires_grad=True)
    teacher = torch.randn_like(z, requires_grad=True)
    hidden = torch.zeros(2, 1, 4, 4, dtype=torch.bool); hidden[:, :, :2] = True
    valid = torch.ones_like(hidden); valid[:, :, 0, 0] = False
    loss, stats = context_relational_loss(z, teacher, hidden, valid, u, a)
    assert stats['aux_selected_cells'] == 14 and stats['context_visible_keys'] == 8
    assert 0 <= stats['context_target_entropy'] <= 1.000001 and stats['context_target_information'] > 0
    loss.backward()
    assert z.grad.abs().sum() > 0 and torch.equal(z.grad[:, :, 0, 0], torch.zeros_like(z.grad[:, :, 0, 0]))
    assert teacher.grad is None and u.grad is None and a.grad is None
    same, _ = context_relational_loss(teacher, teacher, hidden, valid, u, a)
    assert same.abs() < 1e-6
    for mask, validity in ((torch.zeros_like(hidden), valid), (torch.ones_like(hidden), valid), (hidden, torch.zeros_like(valid))):
        inp = z.detach().clone().requires_grad_()
        empty, record = context_relational_loss(inp, teacher, mask, validity, u, a)
        empty.backward()
        assert empty == 0 and record['aux_selected_cells'] == 0 and inp.grad.abs().sum() == 0
    print('Explicit source metric, teacher/source stop-grad, padding and empty contexts OK')


def test_actual_mrm_routing():
    torch.manual_seed(87)
    student, decoder = build_segformer_b2(); decoder.eval()
    teacher, td = copy.deepcopy(student).eval(), copy.deepcopy(decoder).eval()
    source = dict(classifier=td.state_dict())
    settings = json.loads((Path(__file__).resolve().parents[1]/'configs/snd_mrm_context_vh2pd_b2.json').read_text())
    settings.update(objective='context-relation', prepare_updates=1, warmup_updates=1, ramp_updates=1,
                    hidden_dim=64, depth=1, weight=.5)
    aux = RelationalAux(source, settings, 'cpu', 1, 2e-5)
    image = torch.randn(1, 3, 64, 64)
    parameters = tuple(student.parameters()) + tuple(decoder.parameters())
    main = decoder(student(image)[1], (64, 64)).square().mean(); main.backward()
    before = [None if p.grad is None else p.grad.clone() for p in parameters]
    r_before = [p.detach().clone() for p in aux.rebuilder.parameters()]
    rng = torch.get_rng_state().clone()
    aux.backward(1, 2e-5, main, student, decoder, teacher, td, image, image, geometry(64))
    assert torch.equal(rng, torch.get_rng_state()) and aux.last_record['aux_encoder_weight'] == 0
    assert aux.last_record['aux_preparing'] and aux.last_record['aux_rebuilder_grad_norm'] > 0
    for p, g in zip(parameters, before):
        assert p.grad is None if g is None else torch.equal(p.grad, g)
    aux.step()
    assert any(not torch.equal(p, old) for p, old in zip(aux.rebuilder.parameters(), r_before))
    state = copy.deepcopy(aux.state_dict())
    other = RelationalAux(source, settings, 'cpu', 1, 2e-5); other.load_state_dict(state)
    outputs = []
    for module in (aux, other):
        student.zero_grad(); decoder.zero_grad(); torch.manual_seed(95)
        main = decoder(student(image)[1], (64, 64)).square().mean(); main.backward()
        before_e = [None if p.grad is None else p.grad.clone() for p in student.parameters()]
        before_d = [None if p.grad is None else p.grad.clone() for p in decoder.parameters()]
        buffers = {n:b.clone() for n,b in decoder.named_buffers()}
        rng = torch.get_rng_state().clone()
        module.backward(2, 2e-5, main, student, decoder, teacher, td, image, image, geometry(64))
        assert torch.equal(rng, torch.get_rng_state())
        delta_sq = norm_sq = 0
        for p, g in zip(student.parameters(), before_e):
            if p.grad is not None:
                original = torch.zeros_like(p) if g is None else g
                delta_sq += (p.grad-original).square().sum(); norm_sq += original.square().sum()
        ratio = float(delta_sq.sqrt()/norm_sq.sqrt())
        assert 0 < ratio <= .100001 and abs(ratio-module.last_record['aux_grad_ratio']) < 1e-6
        for p, g in zip(decoder.parameters(), before_d):
            assert p.grad is None if g is None else torch.equal(p.grad, g)
        assert all(torch.equal(b, buffers[n]) for n,b in decoder.named_buffers())
        assert all(p.grad is None for p in tuple(teacher.parameters())+tuple(td.parameters()))
        assert module.last_record['rebuilder_update_weight'] == 1
        outputs.append((module.last_record, [p.grad.clone() for p in module.rebuilder.parameters() if p.grad is not None]))
    assert outputs[0][0] == outputs[1][0]
    assert all(torch.equal(a,b) for a,b in zip(outputs[0][1],outputs[1][1]))
    inactive = RelationalAux(source, dict(settings, weight=0), 'cpu', 1, 2e-5)
    rng = torch.get_rng_state().clone()
    inactive.backward(1, 2e-5, torch.tensor(1.), None, None, None, None, None, None, None)
    inactive.step()
    assert inactive.forward_count == 0 and not inactive.optimizer.state and torch.equal(rng, torch.get_rng_state())
    # No main encoder gradient: R learns, encoder receives no update despite nominal weight.
    student.zero_grad(); decoder.zero_grad()
    other.backward(3, 2e-5, torch.tensor(0.), student, decoder, teacher, td, image, image, geometry(64))
    assert other.last_record['aux_encoder_weight'] == 0 and all(p.grad is None for p in parameters)
    assert other.last_record['aux_rebuilder_grad_norm'] > 0
    print('Actual MRM preparation, decoder protection, full-encoder norm cap, zero-main case and exact resume OK')


def test_previous_mrm_exact():
    path = Path(__file__).resolve().parents[1]/'reports/context_remask_20261002/snd_srmp_before.py'
    spec = importlib.util.spec_from_file_location('before_context', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    torch.manual_seed(17)
    initial, head = build_segformer_b2(); head.eval()
    teacher, td = copy.deepcopy(initial).eval(), copy.deepcopy(head).eval()
    settings = dict(objective='relation', rebuilder='mrm', weight=.2, warmup_updates=1, ramp_updates=1,
        prepare_updates=1, soft_kl_weight=.1, temperature=1., mask_ratio=.4, block_size=1,
        hidden_dim=64, depth=1, weight_decay=.01, view_consistency=True, confidence_threshold=.2)
    image = torch.randn(1, 3, 64, 64); results = []
    for klass in (old.RelationalAux, RelationalAux):
        student, decoder = copy.deepcopy(initial), copy.deepcopy(head)
        module = klass(dict(classifier=td.state_dict()), settings, 'cpu', 1, 2e-5)
        records = []
        for step in (1, 2):
            student.zero_grad(); decoder.zero_grad(); torch.manual_seed(501+step)
            main = decoder(student(image)[1], (64, 64)).square().mean(); main.backward()
            module.backward(step, 2e-5, main, student, decoder, teacher, td, image, image, geometry(64))
            records.append(copy.deepcopy(module.last_record)); module.step()
        results.append((records, [p.grad for p in tuple(student.parameters())+tuple(decoder.parameters())],
                        copy.deepcopy(module.state_dict())))
    assert results[0][0] == results[1][0]
    assert all((a is None and b is None) or torch.equal(a,b) for a,b in zip(results[0][1],results[1][1]))
    for key in ('rebuilder',):
        assert all(torch.equal(v, results[1][2][key][n]) for n,v in results[0][2][key].items())
    assert torch.equal(results[0][2]['mask_rng'], results[1][2]['mask_rng'])
    print('Previous gated MRM preparation/soft-KL: exact losses, main gradients, R weights and mask RNG')


def test_screening_configuration():
    from train_SND_remote import configuration
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root/'configs/snd_vh2pd_b2.json').read_text())
    args = SimpleNamespace(device='cpu', source_checkpoint=root/'unused_source.pth', warm_checkpoint=None, warm_stop_updates=6000)
    cfg = configuration(recipe, 'warm', args, root/'unused_test_output')
    assert cfg.SOLVER.STOP_ITER == 6000 and cfg.SOLVER.MAX_ITER == 32000
    assert recipe['warm']['steps'] == 20000 and recipe['warm']['steps']+recipe['snd']['steps'] == 32000
    assert cfg.SOLVER.BATCH_SIZE == 6 and cfg.DTU.EMA_WEIGHT == .99
    print('6000 fixed screening keeps planned warm20000 / LR horizon32000 / batch6 / EMA.99')


if __name__ == '__main__':
    torch.set_num_threads(4)
    test_kernel_and_loss(); test_actual_mrm_routing(); test_previous_mrm_exact(); test_screening_configuration()
    print('Context relation OK')
