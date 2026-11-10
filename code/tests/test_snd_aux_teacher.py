"""Run with python -m tests.test_snd_aux_teacher (torch, CPU)."""
import copy
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

from core.models.segformer import build_segformer_b2
from scripts.snd_srmp import RelationalAux, clean_student_target
from tests.test_snd_aux_improvements import geometry

ROOT = Path(__file__).resolve().parents[1]


def check_eval_target():
    class Encoder(nn.Sequential):
        def forward(self, image):
            return None, super().forward(image)

    class Decoder(nn.Sequential):
        def fuse_features(self, features):
            return self(features)

    student = Encoder(nn.BatchNorm2d(3), nn.Dropout2d(.5)).train()
    decoder = Decoder(nn.BatchNorm2d(3), nn.Dropout2d(.5)).train()
    decoder[0].eval()  # Real trainer freezes BN individually.
    modes = [m.training for model in (student, decoder) for m in model.modules()]
    buffers = [b.clone() for model in (student, decoder) for b in model.buffers()]
    image = torch.randn(2, 3, 8, 8)
    rng = torch.get_rng_state().clone()
    target = clean_student_target(student, decoder, image)
    assert not target.requires_grad and torch.equal(target, clean_student_target(student, decoder, image))
    assert modes == [m.training for model in (student, decoder) for m in model.modules()]
    assert all(torch.equal(a,b) for a,b in zip(buffers, [b for model in (student, decoder) for b in model.buffers()]))
    assert torch.equal(rng, torch.get_rng_state())
    # Even a failed forward must not thaw BN or leave the main student in eval.
    try:
        clean_student_target(student, decoder, torch.randn(1, 4, 8, 8))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Expected invalid channel count')
    assert modes == [m.training for model in (student, decoder) for m in model.modules()]
    print('Clean target: detached, deterministic, RNG/BN safe, mixed modes restored on success and failure')


def check_training():
    torch.manual_seed(83)
    student, decoder = build_segformer_b2()
    decoder.train()
    decoder.linear_fuse[1].eval()
    teacher, td = copy.deepcopy(student).eval(), copy.deepcopy(decoder).eval()
    source = dict(classifier=copy.deepcopy(td.state_dict()))
    settings = json.loads((ROOT/'configs/snd_mrm_source_contrast_vh2pd_b2.json').read_text())
    settings.update(objective='source-contrast', prepare_updates=1, warmup_updates=1,
                    ramp_updates=1, hidden_dim=64, depth=1, contrast_grid=8, contrast_keys_per_class=4)
    image = torch.randn(1, 3, 64, 64)
    spec = importlib.util.spec_from_file_location('before_aux_teacher',
        ROOT/'reports/fast_aux_teacher_20261002/code_before/snd_srmp.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

    def run(aux, step):
        student.zero_grad(); decoder.zero_grad(); torch.manual_seed(99)
        main = decoder(student(image)[1], (64,64)).square().mean()
        main.backward()
        params = tuple(student.parameters()) + tuple(decoder.parameters())
        grads = [None if p.grad is None else p.grad.clone() for p in params]
        modes = [m.training for model in (student,decoder) for m in model.modules()]
        buffers = [b.clone() for model in (student,decoder) for b in model.buffers()]
        rng = torch.get_rng_state().clone()
        aux.backward(step, 2e-5, main, student, decoder, teacher, td, image, image, geometry(64))
        assert torch.equal(rng, torch.get_rng_state())
        assert modes == [m.training for model in (student,decoder) for m in model.modules()]
        assert all(torch.equal(a,b) for a,b in zip(buffers,[b for model in (student,decoder) for b in model.buffers()]))
        decoder_grads = grads[len(tuple(student.parameters())):]
        assert all(p.grad is None if g is None else torch.equal(p.grad,g) for p,g in zip(decoder.parameters(),decoder_grads))
        assert all(p.grad is None for model in (teacher,td) for p in model.parameters())
        assert aux.last_record['aux_grad_ratio'] <= .100001
        record = {k:v for k,v in aux.last_record.items() if not k.startswith('aux_target_')}
        all_grads = [None if p.grad is None else p.grad.clone() for p in params+tuple(aux.rebuilder.parameters())]
        return record, all_grads

    def same(a,b):
        assert a[0] == b[0]
        assert all((x is None and y is None) or torch.equal(x,y) for x,y in zip(a[1],b[1]))

    previous = old.RelationalAux(source, settings, 'cpu', 1, 2e-5)
    ema = RelationalAux(source, dict(settings,target_teacher='ema'), 'cpu', 1, 2e-5)
    fresh = RelationalAux(source, dict(settings,target_teacher='student'), 'cpu', 1, 2e-5)
    a,b,c = run(previous,1),run(ema,1),run(fresh,1)
    same(a,b); same(b,c)
    assert fresh.last_record['aux_target_teacher'] == 'ema'
    for aux in (previous,ema,fresh): aux.step()
    same(run(previous,2),run(ema,2))
    # Make the detached EMA meaningfully different from the current model.
    with torch.no_grad(): td.linear_fuse[0].weight.add_(.02)
    teacher_state = [p.clone() for model in (teacher,td) for p in model.parameters()]
    run(fresh,20)
    assert fresh.last_record['aux_target_teacher'] == 'student'
    assert fresh.last_record['aux_target_ema_cosine'] < .99999
    assert fresh.last_record['aux_grad_ratio'] > 0
    assert all(torch.equal(a,b) for a,b in zip(teacher_state,[p for model in (teacher,td) for p in model.parameters()]))
    fresh.step()
    restored = RelationalAux(source,dict(settings,target_teacher='student'),'cpu',1,2e-5)
    restored.load_state_dict(copy.deepcopy(fresh.state_dict()))
    # Compare the same next step after restoring R, optimizer and private RNG.
    same(run(fresh,21),run(restored,21))
    assert torch.equal(fresh.mask_rng.get_state(), restored.mask_rng.get_state())
    inactive = RelationalAux(source,dict(settings,target_teacher='student',weight=0),'cpu',1,2e-5)
    inactive.backward(1,2e-5,torch.tensor(1.),None,None,None,None,None,None,None)
    assert inactive.forward_count == 0
    try: RelationalAux(source,dict(settings,target_teacher='invalid'),'cpu',1,2e-5)
    except ValueError: pass
    else: raise AssertionError('Invalid teacher accepted')
    try: RelationalAux(source,dict(settings,objective='relation',target_teacher='student'),'cpu',1,2e-5)
    except ValueError: pass
    else: raise AssertionError('Unsupported teacher/objective accepted')
    print('Exact previous EMA path and shared R preparation; fresh routing/cap, unchanged decoder/main teacher, resume and aux0 OK')


if __name__ == '__main__':
    torch.set_num_threads(4)
    check_eval_target()
    check_training()
