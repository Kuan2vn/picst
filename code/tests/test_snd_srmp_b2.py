"""Small math, gradient, mask, and resume checks for the B2 auxiliary."""
import copy
import json
from pathlib import Path

import torch
from torch import nn

from scripts.snd_srmp import RelationalAux, gradient_alignment, source_readout, teacher_site_mask
from train_SRMP import relational_cosine_map


def test_relation_math():
    torch.manual_seed(4)
    source = {'classifier': {'linear_pred.weight': torch.randn(6, 768, 1, 1),
                             'linear_pred.bias': torch.randn(6)}}
    _, _, directions, gram = source_readout(source)
    assert torch.allclose(gram, gram.T, atol=1e-6)
    assert torch.allclose(gram.diag(), torch.ones(6), atol=1e-6)
    logits = torch.randn(2, 6, 3, 4, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(2, 6, 3, 4, dtype=torch.float64).softmax(1)
    p = logits.softmax(1)
    actual = relational_cosine_map(p, teacher, gram).mean()
    embedded_p = torch.einsum('bchw,cd->bdhw', p.float(), directions)
    embedded_q = torch.einsum('bchw,cd->bdhw', teacher.float(), directions)
    explicit = (1 - torch.nn.functional.cosine_similarity(embedded_p, embedded_q, dim=1)).mean()
    assert torch.allclose(actual, explicit, atol=1e-6)
    ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    ge = torch.autograd.grad(explicit, logits)[0]
    assert torch.allclose(ga, ge, atol=1e-6)
    assert not torch.allclose(gram, torch.eye(6))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.patch_embed1 = nn.Module()
        self.backbone.patch_embed1.proj = nn.Conv2d(3, 768, 1)

    def forward(self, image):
        feature = self.backbone.patch_embed1.proj(image)
        return feature, (feature,)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fuse = nn.Sequential(nn.Conv2d(768, 768, 1))
        self.linear_pred = nn.Conv2d(768, 6, 1)

    def fuse_features(self, features):
        return self.linear_fuse(features[0])


def test_aux_backward_and_resume():
    torch.manual_seed(5)
    source = {'classifier': {'linear_pred.weight': torch.randn(6, 768, 1, 1),
                             'linear_pred.bias': torch.randn(6)}}
    settings = dict(objective='relation', hidden_dim=16, depth=1, weight_decay=.01,
                    warmup_updates=0, ramp_updates=1, weight=.1, temperature=1.,
                    mask_ratio=.5, block_size=4)
    student, decoder = Encoder(), Decoder()
    teacher, teacher_decoder = copy.deepcopy(student).eval(), copy.deepcopy(decoder).eval()
    aux = RelationalAux(source, settings, 'cpu', 1, 2e-5)
    image = torch.randn(1, 3, 16, 16)
    params = {'rescale_size': (torch.tensor([16]), torch.tensor([16])),
              'random_crop_axis': tuple(torch.tensor([v]) for v in (0, 16, 0, 16)),
              'RandomHorizontalFlip': torch.tensor([False])}
    main = decoder.fuse_features(student(image)[1]).square().mean()
    main.backward()
    main_head_grad = decoder.linear_pred.weight.grad
    assert main_head_grad is None
    aux.backward(1, 2e-5, main, student, decoder, teacher, teacher_decoder,
                 image, image, params)
    assert aux.last_record['mask_fraction'] > 0
    assert aux.rebuilder.project.weight.grad.abs().sum() > 0
    assert student.backbone.patch_embed1.proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert decoder.linear_pred.weight.grad is None
    assert -1 <= aux.last_record['aux_grad_cosine'] <= 1
    assert aux.last_record['aux_selected_fraction'] == 1
    aux.step()
    state = copy.deepcopy(aux.state_dict())
    other = RelationalAux(source, settings, 'cpu', 1, 2e-5)
    other.load_state_dict(state)
    assert other.forward_count == 1
    assert len(other.optimizer.state) > 0
    assert torch.equal(other.mask_rng.get_state(), aux.mask_rng.get_state())
    assert torch.equal(other.aux_cpu_rng, aux.aux_cpu_rng)
    for original, restored in zip(aux.rebuilder.parameters(), other.rebuilder.parameters()):
        original.grad = torch.full_like(original, .001)
        restored.grad = original.grad.clone()
    other.last_record = aux.last_record
    aux.step()
    other.step()
    assert all(torch.equal(a, b) for a, b in zip(aux.rebuilder.parameters(), other.rebuilder.parameters()))
    hidden = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    hidden[:, :, :8, :8] = True
    features = torch.randn(1, 768, 16, 16)
    changed = features.clone()
    changed[:, :, :8, :8] += 100
    with torch.no_grad():
        assert torch.equal(aux.rebuilder(features, hidden), aux.rebuilder(changed, hidden))


def test_zero_weight_is_inert():
    source = {'classifier': {'linear_pred.weight': torch.randn(6, 768, 1, 1),
                             'linear_pred.bias': torch.randn(6)}}
    settings = dict(objective='relation', hidden_dim=16, depth=1, weight_decay=.01,
                    warmup_updates=500, ramp_updates=1000, weight=.1, temperature=1.,
                    mask_ratio=.5, block_size=4)
    original_rng = torch.get_rng_state().clone()
    aux = RelationalAux(source, settings, 'cpu', 1, 2e-5)
    assert torch.equal(torch.get_rng_state(), original_rng)
    original_mask_rng = aux.mask_rng.get_state().clone()
    aux.backward(1, 2e-5, torch.tensor(1.), None, None, None, None, None, None, None)
    aux.step()
    assert aux.forward_count == 0
    assert torch.equal(aux.mask_rng.get_state(), original_mask_rng)
    assert torch.equal(torch.get_rng_state(), original_rng)


def test_aux_weight_schedule():
    aux = object.__new__(RelationalAux)
    aux.settings = dict(weight=.1, warmup_updates=500, ramp_updates=1000)
    assert [aux.weight_at(i) for i in (500, 1000, 1500)] == [0, .05, .1]
    aux.settings = dict(weight=.3, warmup_updates=0, ramp_updates=500)
    assert aux.weight_at(1) == .0006
    assert aux.weight_at(500) == .3
    settings = json.loads((Path(__file__).resolve().parents[1]/'configs/snd_srmp_vh2pd_b2_max1200.json').read_text())
    aux.settings = settings
    assert settings['warmup_updates'] + settings['ramp_updates'] == 1200
    assert aux.weight_at(500) == 0
    assert aux.weight_at(1200) == settings['weight']


def test_gradient_direction_and_empty_confidence_filter():
    g = [torch.tensor([1., 2.]), torch.tensor([3.])]
    assert gradient_alignment(g, g) == (1., 1.)
    ratio, cosine = gradient_alignment(g, [-x for x in g])
    assert ratio == 1 and cosine == -1
    assert gradient_alignment(g, [torch.zeros_like(x) for x in g]) == (0., None)
    source = {'classifier': {'linear_pred.weight': torch.zeros(6, 768, 1, 1),
                             'linear_pred.bias': torch.zeros(6)}}
    # Equal nonzero directions give uniform teacher probabilities, below threshold.
    source['classifier']['linear_pred.weight'][:] = .01
    settings = dict(objective='relation', hidden_dim=16, depth=1, weight_decay=.01,
                    warmup_updates=0, ramp_updates=1, weight=.6, temperature=1.,
                    mask_ratio=.5, block_size=4, confidence_threshold=.7)
    student, decoder = Encoder(), Decoder()
    teacher, teacher_decoder = copy.deepcopy(student).eval(), copy.deepcopy(decoder).eval()
    aux = RelationalAux(source, settings, 'cpu', 1, 2e-5)
    image = torch.randn(1, 3, 8, 8)
    params = {'rescale_size': (torch.tensor([8]), torch.tensor([8])),
              'random_crop_axis': tuple(torch.tensor([v]) for v in (0, 8, 0, 8)),
              'RandomHorizontalFlip': torch.tensor([False])}
    main = decoder.fuse_features(student(image)[1]).square().mean()
    main.backward()
    before = student.backbone.patch_embed1.proj.weight.grad.clone()
    rebuilder_before = copy.deepcopy(aux.rebuilder.state_dict())
    aux.backward(1, 2e-5, main, student, decoder, teacher, teacher_decoder, image, image, params)
    aux.step()
    assert aux.last_record['aux_selected_cells'] == 0
    assert aux.last_record['aux_selected_fraction'] == 0
    assert aux.last_record['mask_fraction'] == .5
    assert aux.last_record['aux_grad_cosine'] is None
    assert torch.equal(before, student.backbone.patch_embed1.proj.weight.grad)
    assert all(torch.equal(v, rebuilder_before[k]) for k, v in aux.rebuilder.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    # Fully permissive filtering must equal the original loss/gradients/mask RNG.
    initial_student, initial_decoder = copy.deepcopy(student), copy.deepcopy(decoder)
    records = []
    for threshold in (0., .1):
        stu, dec = copy.deepcopy(initial_student), copy.deepcopy(initial_decoder)
        stu.zero_grad(); dec.zero_grad()
        branch = RelationalAux(source, dict(settings, confidence_threshold=threshold), 'cpu', 1, 2e-5)
        loss = dec.fuse_features(stu(image)[1]).square().mean(); loss.backward()
        branch.backward(1, 2e-5, loss, stu, dec, teacher, teacher_decoder, image, image, params)
        records.append((branch.last_record['aux_loss'], stu.backbone.patch_embed1.proj.weight.grad.clone()))
    assert records[0][0] == records[1][0]
    assert torch.equal(records[0][1], records[1][1])


def test_readout_optimizer_groups():
    from train_SND import b2_classifier_parameters
    decoder = Decoder()
    groups = b2_classifier_parameters(decoder, 2e-5, 1)
    assert groups[0]['lr'] == 2e-4 and groups[1]['lr'] == 2e-5
    ids = [id(p) for group in groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(list(decoder.parameters()))
    assert {id(p) for p in groups[1]['params']} == {id(p) for p in decoder.linear_pred.parameters()}
    default = torch.optim.AdamW(b2_classifier_parameters(decoder, 2e-5, 10), lr=2e-4)
    assert len(default.param_groups) == 1  # Existing checkpoints retain their original optimizer layout.
    optimizer = torch.optim.AdamW(b2_classifier_parameters(decoder, 2e-5, 0), lr=2e-4)
    original_head = decoder.linear_pred.weight.detach().clone()
    original_fusion = decoder.linear_fuse[0].weight.detach().clone()
    for p in decoder.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    assert torch.equal(original_head, decoder.linear_pred.weight)
    assert not torch.equal(original_fusion, decoder.linear_fuse[0].weight)
    restored = torch.optim.AdamW(b2_classifier_parameters(decoder, 2e-5, 0), lr=2e-4)
    restored.load_state_dict(optimizer.state_dict())
    assert [g['lr_mult'] for g in restored.param_groups] == [10, 0]


def test_view_gate_geometry_and_gradients():
    from scripts.snd_remote import align_full
    from train_SRMP import align_full_image
    from torch.nn import functional as F
    q = torch.tensor([[[[.95, .6, .05, .05]], [[.05, .4, .95, .95]]]], requires_grad=True)
    full = torch.tensor([[[[.95, .6, .95, .05]], [[.05, .4, .05, .95]]]], requires_grad=True)
    sites = torch.tensor([[[[True, True, True, False]]]])
    selected = teacher_site_mask(q, sites, .9, full)
    assert selected.tolist() == [[[[True, False, False, False]]]]
    assert not selected.requires_grad and q.grad is None and full.grad is None
    assert torch.equal(teacher_site_mask(q, sites), sites)
    try: teacher_site_mask(q, sites, .9, full[..., :2])
    except ValueError: pass
    else: raise AssertionError('Mismatched teacher grids accepted')

    torch.manual_seed(51)
    student, decoder = Encoder(), Decoder()
    teacher, td = copy.deepcopy(student).eval(), copy.deepcopy(decoder).eval()
    source = {'classifier': td.state_dict()}
    settings = dict(objective='relation', hidden_dim=16, depth=1, weight_decay=.01,
        warmup_updates=0, ramp_updates=1, weight=.6, temperature=1., mask_ratio=.5,
        block_size=4, view_consistency=True, confidence_threshold=0.)
    aux = RelationalAux(source, settings, 'cpu', 1, 2e-5)
    image = torch.randn(1, 3, 16, 16)
    params = dict(rescale_size=(torch.tensor([24]), torch.tensor([24])),
        padding=tuple(torch.tensor([0]) for _ in range(4)),
        random_crop_axis=tuple(torch.tensor([v]) for v in (3, 19, 5, 21)),
        RandomHorizontalFlip=torch.tensor([True]))
    crop, _ = align_full_image(image, params, (16,16))
    with torch.no_grad():
        full_features = teacher(image)[1]
        q = F.conv2d(td.fuse_features(teacher(crop)[1]), aux.weight, aux.bias).softmax(1)
        native = F.conv2d(td.fuse_features(full_features), aux.weight, aux.bias)
        expected_full = align_full(native, params, align_corners=False).softmax(1)
    # A supplied full-image cache must prevent a second encoder forward.
    old_forward = teacher.forward
    teacher.forward = lambda _: (_ for _ in ()).throw(AssertionError('Cache not reused'))
    actual, record = aux.select_teacher_sites(q, torch.ones_like(q[:,:1],dtype=torch.bool),
        teacher, td, image, params, full_features, diagnostics=True)
    teacher.forward = old_forward
    assert torch.equal(actual, teacher_site_mask(q, torch.ones_like(actual), 0., expected_full))
    assert 0 <= record['aux_view_disagreement'] <= 1
    assert sum(record['aux_selected_class_pixels']) == int(actual.sum())

    # Identity geometry is permissive and retains gradients to encoder/R, never to teacher/head.
    params.update(rescale_size=(torch.tensor([16]),torch.tensor([16])),
        random_crop_axis=tuple(torch.tensor([v]) for v in (0,16,0,16)),
        RandomHorizontalFlip=torch.tensor([False]))
    main = decoder.fuse_features(student(image)[1]).square().mean(); main.backward()
    aux.backward(1, 2e-5, main, student, decoder, teacher, td, image, image, params)
    assert aux.last_record['aux_selected_fraction'] == 1
    assert aux.rebuilder.project.weight.grad.abs().sum() > 0
    assert student.backbone.patch_embed1.proj.weight.grad.abs().sum() > 0
    assert decoder.linear_pred.weight.grad is None and all(p.grad is None for p in teacher.parameters())
    aux.step()
    # With no eligible sites, even AdamW momentum/decay must not update R.
    aux.settings['confidence_threshold'] = 1.
    before = copy.deepcopy(aux.rebuilder.state_dict())
    student.zero_grad(); decoder.zero_grad()
    main = decoder.fuse_features(student(image)[1]).square().mean(); main.backward()
    grad_before = student.backbone.patch_embed1.proj.weight.grad.clone()
    aux.backward(20,2e-5,main,student,decoder,teacher,td,image,image,params); aux.step()
    assert aux.last_record['aux_selected_cells'] == 0 and aux.last_record['aux_loss'] == 0
    assert torch.equal(grad_before,student.backbone.patch_embed1.proj.weight.grad)
    assert all(torch.equal(v,before[k]) for k,v in aux.rebuilder.state_dict().items())
    inactive = RelationalAux(source, dict(settings,weight=0), 'cpu',1,2e-5)
    rng = torch.get_rng_state().clone()
    inactive.backward(20,2e-5,torch.tensor(1.),None,None,None,None,None,None,None)
    assert inactive.forward_count == 0 and torch.equal(rng,torch.get_rng_state())


if __name__ == '__main__':
    torch.set_num_threads(4)
    test_relation_math()
    test_aux_backward_and_resume()
    test_zero_weight_is_inert()
    test_aux_weight_schedule()
    test_gradient_direction_and_empty_confidence_filter()
    test_readout_optimizer_groups()
    test_view_gate_geometry_and_gradients()
    print('SND-SRMP relation, gradient, mask and resume OK')
