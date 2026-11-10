"""Run with python -m tests.test_snd_source_contrast (torch, CPU)."""
import copy
import importlib.util
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from core.models.segformer import build_segformer_b2
from scripts.snd_srmp import RelationalAux, masked_source_contrastive_loss
from tests.test_snd_aux_improvements import geometry

ROOT = Path(__file__).resolve().parents[1]


def test_loss():
    teacher = F.one_hot(torch.tensor([[[0, 1], [2, 3]]]), 6).permute(0,3,1,2).float().requires_grad_()
    source = (10*torch.eye(6)).reshape(6,6,1,1).requires_grad_()
    bias = torch.zeros(6, requires_grad=True); gram = torch.eye(6, requires_grad=True)
    hidden = torch.tensor([[[[True, True], [False, False]]]])
    valid = torch.ones_like(hidden)
    def run(student, t=teacher, v=valid, g=gram):
        return masked_source_contrastive_loss(student, t, hidden, v, source, bias, g,
            grid=2, keys_per_class=1, generator=torch.Generator().manual_seed(17))
    matched = teacher.detach().clone().requires_grad_()
    wrong = teacher.detach().flip(-1).clone().requires_grad_()
    good, sg = run(matched); bad, sb = run(wrong)
    assert good < bad and torch.isfinite(good+bad) and sg['contrast_positive_margin'] > sb['contrast_positive_margin']
    bad.backward()
    assert wrong.grad.abs().sum() > 0 and teacher.grad is None and source.grad is None and bias.grad is None and gram.grad is None
    assert max(sb['contrast_keys_by_class']) <= 1 and sb['aux_selected_cells'] == 2
    valid[...,0,0] = False
    inp = wrong.detach().clone().requires_grad_(); loss, record = run(inp); loss.backward()
    assert record['aux_selected_cells'] == 1 and inp.grad[...,0,0].abs().sum() == 0
    same_teacher = torch.zeros_like(teacher).requires_grad_()
    inp = wrong.detach().clone().requires_grad_(); empty, s = run(inp, t=same_teacher)
    empty.backward(); assert empty == 0 and s['aux_selected_cells'] == 0 and inp.grad.abs().sum() == 0
    inp = wrong.detach().clone().requires_grad_(); empty, s = run(inp, v=torch.zeros_like(valid))
    empty.backward(); assert empty == 0 and s['contrast_keys'] == 0
    # Gram affects negative weights, without introducing teacher/source gradients.
    correlated = .8*torch.ones(6,6)+.2*torch.eye(6)
    _, original = run(wrong, v=torch.ones_like(valid)); _, changed = run(wrong, v=torch.ones_like(valid), g=correlated)
    assert changed['contrast_negative_mass'] < original['contrast_negative_mass']
    try: masked_source_contrastive_loss(wrong, teacher, hidden, valid, source, bias, gram, grid=0)
    except ValueError: pass
    else: raise AssertionError('Invalid grid accepted')
    print('Paired positive ranking, source CRS influence, finite wrong-match gradients, padding, empty negatives and stop-grad OK')


def test_routing_resume_and_old_context():
    torch.manual_seed(83)
    encoder, decoder = build_segformer_b2(); decoder.eval()
    teacher, td = copy.deepcopy(encoder).eval(), copy.deepcopy(decoder).eval()
    source = dict(classifier=td.state_dict())
    settings = json.loads((ROOT/'configs/snd_mrm_source_contrast_vh2pd_b2.json').read_text())
    settings.update(objective='source-contrast', prepare_updates=1, warmup_updates=1, ramp_updates=1,
                    hidden_dim=64, depth=1, contrast_grid=8, contrast_keys_per_class=4, weight=.5)
    aux = RelationalAux(source, settings, 'cpu', 1, 2e-5)
    image = torch.randn(1,3,64,64)
    main = decoder(encoder(image)[1], (64,64)).square().mean(); main.backward()
    params = tuple(encoder.parameters()) + tuple(decoder.parameters())
    before = [None if p.grad is None else p.grad.clone() for p in params]
    rng = torch.get_rng_state().clone(); r_before = [p.detach().clone() for p in aux.rebuilder.parameters()]
    aux.backward(1, 2e-5, main, encoder, decoder, teacher, td, image, image, geometry(64)); aux.step()
    assert torch.equal(rng, torch.get_rng_state()) and aux.last_record['aux_preparing']
    assert all(p.grad is None if g is None else torch.equal(p.grad,g) for p,g in zip(params,before))
    assert any(not torch.equal(p,b) for p,b in zip(aux.rebuilder.parameters(),r_before))
    state = copy.deepcopy(aux.state_dict())
    other = RelationalAux(source, settings, 'cpu', 1, 2e-5); other.load_state_dict(state)
    records=[]
    for module in (aux,other):
        encoder.zero_grad(); decoder.zero_grad(); torch.manual_seed(99)
        main = decoder(encoder(image)[1], (64,64)).square().mean(); main.backward()
        eg = [p.grad.clone() for p in encoder.parameters()]
        dg = [None if p.grad is None else p.grad.clone() for p in decoder.parameters()]
        buffers = {n:b.clone() for n,b in decoder.named_buffers()}
        rng = torch.get_rng_state().clone()
        module.backward(2, 2e-5, main, encoder, decoder, teacher, td, image, image, geometry(64))
        assert torch.equal(rng, torch.get_rng_state())
        assert 0 < module.last_record['contrast_negative_mass'] <= module.last_record['contrast_source_negative_mass'] + 1e-6
        ratio = sum((p.grad-g).square().sum() for p,g in zip(encoder.parameters(),eg)).sqrt()/sum(g.square().sum() for g in eg).sqrt()
        assert 0 < ratio <= .100001 and abs(float(ratio)-module.last_record['aux_grad_ratio']) < 1e-6
        assert all(p.grad is None if g is None else torch.equal(p.grad,g) for p,g in zip(decoder.parameters(),dg))
        assert all(torch.equal(b,buffers[n]) for n,b in decoder.named_buffers())
        assert all(p.grad is None for p in tuple(teacher.parameters())+tuple(td.parameters()))
        records.append((copy.deepcopy(module.last_record),[p.grad.clone() for p in module.rebuilder.parameters() if p.grad is not None]))
    assert records[0][0] == records[1][0] and all(torch.equal(a,b) for a,b in zip(records[0][1],records[1][1]))
    inactive = RelationalAux(source,dict(settings,weight=0),'cpu',1,2e-5)
    inactive.backward(1,2e-5,torch.tensor(1.),None,None,None,None,None,None,None); inactive.step()
    assert inactive.forward_count == 0 and not inactive.optimizer.state
    print('MRM R-only preparation, full-encoder cap, decoder preservation, private RNG, exact resume and aux0 OK')
    # Verify the existing context objective against the exact pre-change helper.
    path=ROOT/'reports/semantic_recovery_20261002/code_before/snd_srmp.py'
    spec=importlib.util.spec_from_file_location('before_contrast',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    config=json.loads((ROOT/'configs/snd_mrm_context_vh2pd_b2.json').read_text())
    config.update(objective='context-relation', prepare_updates=0, warmup_updates=0,ramp_updates=1,hidden_dim=64,depth=1)
    results=[]
    for cls in (old.RelationalAux,RelationalAux):
        encoder.zero_grad();decoder.zero_grad();torch.manual_seed(67)
        module=cls(source,config,'cpu',1,2e-5)
        main=decoder(encoder(image)[1],(64,64)).square().mean();main.backward()
        module.backward(1,2e-5,main,encoder,decoder,teacher,td,image,image,geometry(64))
        results.append((module.last_record,[None if p.grad is None else p.grad.clone() for p in params]))
    assert results[0][0] == results[1][0]
    assert all((a is None and b is None) or torch.equal(a,b) for a,b in zip(results[0][1],results[1][1]))
    print('Previous context objective: exact numerical parity')


if __name__ == '__main__':
    torch.set_num_threads(4)
    test_loss(); test_routing_resume_and_old_context()
