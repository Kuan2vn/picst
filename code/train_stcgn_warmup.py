"""St-CGN warm-up stage (re-implementation; St-CGN, IEEE TGRS 64, 2026, Sec. III-B and III-E, Eqs. 1-10, 24-25).

Source-side stage that runs BEFORE adaptation, on the labelled source split, starting from the shared source40k
checkpoint (which plays the role of the paper's source-pretrained teacher):
  L_wp = L_src + lambda_inst L_inst + (1 - lambda_inst) L_reg + L_kpd + L_kfd
  * L_src : pixel CE on the student view (source labels).
  * L_inst: class-masked prototypes of student / teacher projected features on two independently augmented
            views (flip, 90-degree rotation, Gaussian blur, colour jitter, grayscale; SimCLR-style pool),
            InfoNCE over the classes present in both views (Eq. 3-4).
  * L_reg : region matching: random teacher anchors, positive = most similar student pixel anywhere in the
            student view, accepted when the similarity exceeds a curriculum threshold
            eps_t = min(eps_max, max(eps_{t-1}, s_min)), eps_0 = 0.7, eps_max = 0.9, 255 random negatives (Eq. 5-8).
  * L_kpd / L_kfd: teacher -> student KL on predictions and MSE on decoder features (same student view).
  Teacher = EMA of the student, alpha = 0.996. Anchors (Eq. 1) = class-masked means of the teacher's decoder
  features over all source images; they initialise the CbPCM prototypes of the ST stage.
Unspecified in the paper and fixed here (disclosed): lambda_inst = 0.3 (paper: lower instance weight is better;
lambda_reg = 1 - lambda_inst), InfoNCE temperature 0.1, 256 anchors per image, projection MLP 768->768->256,
KD weights 1, 2000 warm-up iterations (paper: 2000 student warm-up iterations after 3000 teacher pre-training
iterations, the latter replaced by the shared source40k), AdamW with the source recipe (6e-5, head x10, linear
decay), batch 6, crop 512. The output checkpoint holds the EMA teacher weights and source-checkpoint metadata.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from train_SRMP_remote import B2_PROTOCOL, load_checkpoint, make_model, sha256_file

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ARCH = 'segformer_mit_b2_nvidia_mlp768'


def view(image, label, rng):
    k = rng.randrange(4)
    if rng.random() < .5:
        image, label = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if k:
        image, label = image.rotate(90 * k), label.rotate(90 * k)
    if rng.random() < .8:
        image = TF.adjust_brightness(image, rng.uniform(.6, 1.4))
        image = TF.adjust_contrast(image, rng.uniform(.6, 1.4))
        image = TF.adjust_saturation(image, rng.uniform(.6, 1.4))
        image = TF.adjust_hue(image, rng.uniform(-.1, .1))
    if rng.random() < .2:
        image = TF.to_grayscale(image, 3)
    if rng.random() < .5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, 2.)))
    return TF.normalize(TF.to_tensor(image), MEAN, STD), torch.from_numpy(np.array(label, dtype=np.int64))


class SourceViews(Dataset):
    def __init__(self, root, steps, batch, seed, two_views=True):
        with (Path(root)/'manifests/source_train.csv').open(newline='', encoding='utf-8') as handle:
            self.rows = [(Path(root)/r['image'], Path(root)/r['label']) for r in csv.DictReader(handle)]
        self.two_views, self.seed = two_views, seed
        rng = random.Random(seed)
        self.order = [rng.randrange(len(self.rows)) for _ in range(steps * batch)] if steps else list(range(len(self.rows)))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        image_path, label_path = self.rows[self.order[index]]
        with Image.open(image_path) as im, Image.open(label_path) as lb:
            image, label = im.convert('RGB'), lb.copy()
        if not self.two_views:
            return TF.normalize(TF.to_tensor(image), MEAN, STD), torch.from_numpy(np.array(label, dtype=np.int64))
        rng = random.Random(self.seed * 1000003 + index)
        return view(image, label, rng) + view(image, label, rng)


class MLP(nn.Module):
    def __init__(self, dim=768, out=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.ReLU(inplace=True), nn.Conv2d(dim, out, 1))

    def forward(self, x):
        return self.net(x)


def forward(encoder, classifier, x):
    fused = classifier.fuse_features(encoder(x)[1])
    return fused, classifier.classify_fused(fused, x.shape[-2:])


def small_labels(label, size):
    return F.interpolate(label[:, None].float(), size=size, mode='nearest')[:, 0].long()


def instance_loss(ps, pt, ys, yt, tau):
    """Eq. 3-4: class prototypes of the student/teacher views, InfoNCE over classes present in both."""
    ps, pt = F.normalize(ps, dim=1), F.normalize(pt, dim=1)
    losses = []
    for i in range(ps.shape[0]):
        classes = [c for c in range(6) if (ys[i] == c).any() and (yt[i] == c).any()]
        if not classes:
            continue
        zs = torch.stack([F.normalize((ps[i] * (ys[i] == c)).sum((1, 2)) / (ys[i] == c).sum(), dim=0) for c in classes])
        zt = torch.stack([F.normalize((pt[i] * (yt[i] == c)).sum((1, 2)) / (yt[i] == c).sum(), dim=0) for c in classes])
        logits = zs @ zt.T / tau
        losses.append(F.cross_entropy(logits, torch.arange(len(classes), device=logits.device)))
    return torch.stack(losses).mean() if losses else ps.sum() * 0


def region_loss(fs, ft, eps, tau, anchors=256, negatives=255):
    """Eq. 5-8: global matching of teacher anchors to the student view with a curriculum similarity threshold."""
    b, d = fs.shape[:2]
    vs = F.normalize(fs.view(b, d, -1), dim=1); vt = F.normalize(ft.view(b, d, -1), dim=1)
    n = vs.shape[-1]
    losses, minima, accepted = [], [], 0
    for i in range(b):
        idx = torch.randperm(n, device=fs.device)[:anchors]
        a = vt[i][:, idx].T                       # A x D
        sim = a @ vs[i]                            # A x N
        best, pos = sim.detach().max(1)
        valid = best > eps
        if not valid.any():
            continue
        accepted += int(valid.sum()); minima.append(best[valid].min())
        neg = torch.randint(0, n - 1, (anchors, negatives), device=fs.device)
        neg = neg + (neg >= pos[:, None]).long()   # exclude the positive index
        logits = torch.cat([sim.gather(1, pos[:, None]), sim.gather(1, neg)], 1)[valid] / tau
        losses.append(F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=fs.device)))
    loss = torch.stack(losses).mean() if losses else fs.sum() * 0
    s_min = float(torch.stack(minima).min()) if minima else None
    return loss, s_min, accepted


@torch.no_grad()
def ema_update(teacher, student, alpha):
    for t, s in zip(teacher.state_dict().values(), student.state_dict().values()):
        if t.dtype.is_floating_point:
            t.mul_(alpha).add_(s.detach(), alpha=1 - alpha)
        else:
            t.copy_(s)


@torch.no_grad()
def anchors(encoder, classifier, root, device, workers):
    loader = DataLoader(SourceViews(root, 0, 1, 0, two_views=False), batch_size=8, num_workers=workers)
    sums = torch.zeros(6, 768, device=device); counts = torch.zeros(6, device=device)
    encoder.eval(); classifier.eval()
    for image, label in loader:
        image, label = image.to(device), label.to(device)
        fused, _ = forward(encoder, classifier, image)
        y = small_labels(label, fused.shape[-2:])
        f = fused.permute(0, 2, 3, 1)
        for c in range(6):
            m = y == c
            if m.any():
                sums[c] += f[m].sum(0); counts[c] += m.sum()
    return sums / counts.clamp_min(1)[:, None], counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--source-checkpoint', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, required=True, help='data root holding manifests/source_train.csv + source images')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=6e-5); ap.add_argument('--head-lr-mult', type=float, default=10.)
    ap.add_argument('--lambda-inst', type=float, default=.3); ap.add_argument('--tau', type=float, default=.1)
    ap.add_argument('--ema', type=float, default=.996); ap.add_argument('--eps0', type=float, default=.7)
    ap.add_argument('--eps-max', type=float, default=.9); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    final = a.output/'stcgn_warmup.pth'
    if final.is_file() and (a.output/'stcgn_prototypes.pt').is_file():
        print('Warm-up already complete:', final); return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    source = load_checkpoint(a.source_checkpoint)
    if source.get('architecture') != ARCH or source.get('stage') != 'source':
        raise ValueError('Expected the shared B2 source checkpoint')
    encoder, classifier = make_model(6, architecture=ARCH)
    encoder.load_state_dict(source['feature_extractor']); classifier.load_state_dict(source['classifier'])
    encoder, classifier = encoder.to(device), classifier.to(device)
    t_enc, t_cls = copy.deepcopy(encoder).eval(), copy.deepcopy(classifier).eval()
    for p in list(t_enc.parameters()) + list(t_cls.parameters()):
        p.requires_grad_(False)
    proj = MLP().to(device)
    optimizer = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': a.lr, 'base': a.lr},
                                   {'params': list(classifier.parameters()) + list(proj.parameters()),
                                    'lr': a.lr * a.head_lr_mult, 'base': a.lr * a.head_lr_mult}], weight_decay=.01)
    loader = DataLoader(SourceViews(a.source_root, a.steps, a.batch, a.seed), batch_size=a.batch,
                        num_workers=a.workers, drop_last=True, pin_memory=device.type == 'cuda')
    eps, log = a.eps0, (a.output/'warmup_history.jsonl').open('a', encoding='utf-8')
    encoder.train(); classifier.train(); proj.train(); start = time.time()
    for step, (x_t, y_t, x_s, y_s) in enumerate(loader, 1):
        for g in optimizer.param_groups:
            g['lr'] = g['base'] * (1 - (step - 1) / a.steps)
        x_t, y_t, x_s, y_s = (v.to(device, non_blocking=True) for v in (x_t, y_t, x_s, y_s))
        fused_s, logits_s = forward(encoder, classifier, x_s)
        with torch.no_grad():
            fused_t, _ = forward(t_enc, t_cls, x_t)          # teacher view t
            fused_ts, logits_ts = forward(t_enc, t_cls, x_s)  # teacher on the student view (distillation)
        proj_s, proj_t = proj(fused_s), proj(fused_t).detach()
        size = fused_s.shape[-2:]
        l_src = F.cross_entropy(logits_s, y_s, ignore_index=255)
        l_inst = instance_loss(proj_s, proj_t, small_labels(y_s, size), small_labels(y_t, size), a.tau)
        l_reg, s_min, accepted = region_loss(proj_s, proj_t, eps, a.tau)
        if s_min is not None:
            eps = min(a.eps_max, max(eps, s_min))
        l_kpd = F.kl_div(F.log_softmax(logits_s, 1), F.softmax(logits_ts, 1), reduction='none').sum(1).mean()
        l_kfd = F.mse_loss(fused_s, fused_ts)
        loss = l_src + a.lambda_inst * l_inst + (1 - a.lambda_inst) * l_reg + l_kpd + l_kfd
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite St-CGN warm-up loss at step {step}')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        ema_update(t_enc, encoder, a.ema); ema_update(t_cls, classifier, a.ema)
        if step % 50 == 0 or step == 1:
            record = dict(step=step, loss=float(loss), src=float(l_src), inst=float(l_inst), reg=float(l_reg),
                          kpd=float(l_kpd), kfd=float(l_kfd), eps=eps, accepted_anchors=accepted,
                          lr=optimizer.param_groups[0]['lr'], seconds=round(time.time() - start, 1))
            log.write(json.dumps(record) + '\n'); log.flush(); print(json.dumps(record), flush=True)
    prototypes, counts = anchors(t_enc, t_cls, a.source_root, device, a.workers)
    torch.save({'prototypes': prototypes.cpu(), 'counts': counts.cpu(), 'feature': 'teacher decoder fused 768-d (1/4)',
                'source': 'source_train (labelled), Eq. 1'}, a.output/'stcgn_prototypes.pt')
    state = {k: v for k, v in source.items() if k not in ('feature_extractor', 'classifier', 'optimizer')}
    state.update(feature_extractor=t_enc.state_dict(), classifier=t_cls.state_dict(), iteration=a.steps,
                 training_settings={'steps': a.steps, 'lr': a.lr, 'head_lr_mult': a.head_lr_mult, 'amp': False},
                 checkpoint_selection='last_scheduled', protocol=B2_PROTOCOL,
                 stcgn_warmup={'parent_source_sha256': sha256_file(a.source_checkpoint), 'steps': a.steps,
                               'batch': a.batch, 'lambda_inst': a.lambda_inst, 'tau': a.tau, 'ema': a.ema,
                               'eps0': a.eps0, 'eps_max': a.eps_max, 'final_eps': eps, 'seed': a.seed,
                               'weights': 'EMA teacher'})
    state.pop('source_checkpoint_sha256', None)
    torch.save(state, a.output/'stcgn_warmup.pth.tmp'); (a.output/'stcgn_warmup.pth.tmp').replace(final)
    (a.output/'WARMUP_COMPLETE.json').write_text(json.dumps(dict(
        checkpoint_sha256=sha256_file(final), prototypes_sha256=sha256_file(a.output/'stcgn_prototypes.pt'),
        parent_source_sha256=state['stcgn_warmup']['parent_source_sha256'], steps=a.steps,
        prototype_counts=counts.cpu().tolist()), indent=2))
    print('St-CGN warm-up complete:', final)


if __name__ == '__main__':
    main()
