r"""Source-Relational Masked Prediction (SRMP), GTA5 -> Cityscapes.

Implements the final formulation in
``Idea For latent space ReMask (From MAE to MRM).txt``:

    L = L_PL + lambda_rel * (1 - cos(p_rebuilt @ W_s, stopgrad(p_teacher @ W_s)))

The source ASPP classifier is shared and frozen, including its biases. Only the
student encoder and spatial rebuilder are optimized. The clean teacher encoder
is an EMA of the student. The relational loss uses masked, geometrically aligned,
non-copy-paste feature positions. It has no hard pseudo-label supervision.

Framework: the same core model, ReMask reliability-weighted CE, target loader,
CTR/CTR_O refresh, polynomial LR, and Cityscapes evaluation as the g2c20k array.
This is a single-GPU trainer, matching that SLURM job. It does not implement DTU
meta optimization or ReMask's prototype contrastive loss. The inherited dataset
and candidate initializer still open target GT files for diagnostics; those
labels never enter this trainer's optimization objective.

DeepLabV2 has four dilated 3x3 classifier branches, not a C x D linear head.
W_s therefore contains the normalized *effective ASPP kernels* in the union of
their spatial offsets (overlapping centers are summed). This is an explicit
extension of CREL's linear-classifier geometry to a convolutional classifier.
The small Gram matrix A_s = W_s W_s.T evaluates the cosine exactly, without
materializing a huge probability-weighted kernel at every pixel. Using A_s @ p
as the embedding instead would induce A_s squared and a different objective.

Example (run from the repository root; optional flags precede config overrides):
    python train_SRMP.py -cfg configs/deeplabv2_r101_SND_G2C.yaml \
        resume pretrained/G2C_model_iter020000.pth \
        SOLVER.MAX_ITER 30000 SOLVER.STOP_ITER 30000 \
        OUTPUT_DIR results_gta5_to_cityscapes/srmp

Controls: SRMP.OBJECTIVE = relation (default), probability (A=I), masked_ce,
or none (the same frozen-head PL baseline). These are controlled SRMP ablations,
not reproductions of the complete MRM or CREL methods. --self-test needs only
PyTorch and exercises the loss, masking, alignment, gradients, and EMA on CPU.
SRMP_DIAG adds read-only error/reliability and frozen-DINO region audits; it does
not train direction 1 or contribute any ground truth/reference features to loss.
"""

import argparse
import copy
import datetime
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def add_srmp_config(config):
    from yacs.config import CfgNode as CN
    from srmp_diagnostics import add_diagnostic_config

    config.SRMP = CN()
    config.SRMP.OBJECTIVE = "relation"
    config.SRMP.WEIGHT = 0.1
    config.SRMP.TEMPERATURE = 1.0
    config.SRMP.MASK_RATIO = 0.5
    config.SRMP.BLOCK_SIZE = 8  # Feature cells, approximately 64 image pixels.
    config.SRMP.HIDDEN_DIM = 256
    config.SRMP.DEPTH = 4
    config.SRMP.PREDICTOR_LR_MULT = 1.0
    config.SRMP.WARMUP_ITERS = 500
    config.SRMP.RAMP_ITERS = 1000
    config.SRMP.EXCLUDE_HALO = 1  # Feature cells around pasted/padded regions.
    config.SRMP.MAIN_LOSS = "reliability"  # Or "ce" / "snd".
    config.SRMP.GRAD_CLIP = 5.0
    add_diagnostic_config(config)


@torch.no_grad()
def source_kernel_geometry(classifier):
    """Return normalized class kernels W and Gram A for a summed ASPP head.

    Each column block refers to one input channel at one spatial offset. We
    merge weights at equal offsets before computing relationships, because the
    ASPP forward sums its branch logits. Bias is frozen in g_s but, as in CREL,
    is not part of the prototype normalization.
    """
    branches = getattr(classifier, "conv2d_list", None)
    if branches is None or len(branches) == 0:
        raise TypeError("SRMP requires this repository's ASPP_Classifier_V2.")
    offsets = {}
    first = branches[0]
    for conv in branches:
        if not isinstance(conv, nn.Conv2d) or conv.groups != 1:
            raise ValueError("ASPP branches must be ungrouped Conv2d layers.")
        if (conv.in_channels, conv.out_channels) != (first.in_channels, first.out_channels):
            raise ValueError("ASPP branches must share their input/output dimensions.")
        if conv.stride != (1, 1) or conv.padding_mode != "zeros":
            raise ValueError("SRMP geometry expects stride-one, zero-padded ASPP.")
        weight = conv.weight.detach().float().cpu()
        for ky in range(weight.shape[2]):
            for kx in range(weight.shape[3]):
                offset = (ky * conv.dilation[0] - conv.padding[0],
                          kx * conv.dilation[1] - conv.padding[1])
                part = weight[:, :, ky, kx]
                offsets[offset] = offsets[offset] + part if offset in offsets else part.clone()
    kernels = torch.cat([offsets[key] for key in sorted(offsets)], dim=1)
    norms = kernels.norm(dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Source classifier contains non-finite or zero class kernels.")
    directions = kernels / norms
    gram = directions @ directions.T
    return directions, (gram + gram.T) * 0.5


def relational_cosine_map(predicted_prob, teacher_prob, gram):
    """Exactly 1-cos(p W, q W) using A=W W.T; teacher/geometry are detached."""
    p = predicted_prob.float()
    q = teacher_prob.detach().float()
    matrix = gram.detach().to(device=p.device, dtype=torch.float32)
    if p.shape != q.shape or matrix.shape != (p.shape[1], p.shape[1]):
        raise ValueError("Probability grids and source geometry have incompatible shapes.")
    ap = torch.einsum("cd,bdhw->bchw", matrix, p)
    aq = torch.einsum("cd,bdhw->bchw", matrix, q)
    numerator = (p * aq).sum(dim=1)
    norm_p = (p * ap).sum(dim=1).clamp_min(1e-12).sqrt()
    norm_q = (q * aq).sum(dim=1).clamp_min(1e-12).sqrt()
    return 1.0 - (numerator / (norm_p * norm_q)).clamp(-1.0, 1.0)


def masked_mean(values, valid):
    """An empty selection returns a differentiable zero, never NaN."""
    weights = valid.detach().to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def sample_block_mask(batch, height, width, ratio, block_size, device, generator=None):
    if not 0.0 <= ratio < 1.0 or block_size < 1:
        raise ValueError("Require 0 <= mask ratio < 1 and block size >= 1.")
    if ratio == 0 or height * width == 1:
        return torch.zeros(batch, 1, height, width, device=device, dtype=torch.bool)
    gh, gw = math.ceil(height / block_size), math.ceil(width / block_size)
    # Tiny smoke-test grids must still contain both context and prediction sites.
    if gh * gw == 1:
        block_size, gh, gw = 1, height, width
    count = min(gh * gw - 1, max(1, round(gh * gw * ratio)))
    ids = torch.rand(batch, gh * gw, device=device, generator=generator).argsort(dim=1)[:, :count]
    coarse = torch.zeros(batch, gh * gw, device=device, dtype=torch.bool)
    coarse.scatter_(1, ids, True)
    return coarse.view(batch, 1, gh, gw).repeat_interleave(
        block_size, 2
    ).repeat_interleave(block_size, 3)[:, :, :height, :width]


class SpatialBlock(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim),
            nn.GroupNorm(math.gcd(16, dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class SpatialRebuilder(nn.Module):
    """Keep the native feature grid and ASPP dilations; mask before mixing.

    No skip connection carries original hidden tokens to the output. Dilated
    spatial blocks allow context prediction; a pointwise MLP alone cannot infer
    different masked positions from a shared mask token.
    """
    def __init__(self, input_dim, hidden_dim=256, depth=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, input_dim, 1, 1))
        self.project = nn.Conv2d(input_dim, hidden_dim, 1)
        self.context = nn.Sequential(*[
            SpatialBlock(hidden_dim, 2 ** min(i, 3)) for i in range(depth)
        ])
        self.decode = nn.Sequential(nn.GELU(), nn.Conv2d(hidden_dim, input_dim, 1))

    def forward(self, features, hidden):
        masked = torch.where(hidden.bool(), self.mask_token.to(features.dtype), features)
        return self.decode(self.context(self.project(masked)))


def align_full_image(tensor, params, output_size):
    """Replay dataset resize/crop/flip at image resolution, including padding.

    Applying these operations to the clean RGB tensor *before* teacher encoding
    avoids assuming that resizing CNN features is equivalent to encoding a crop.
    Also used on full-image teacher logits for the inherited main PL objective.
    """
    aligned, validity = [], []
    out_h, out_w = output_size
    for i in range(tensor.shape[0]):
        if "rescale_size" in params:
            h, w = (int(v[i]) for v in params["rescale_size"])
        elif "resize" in params:
            h, w = (int(v[i]) for v in params["resize"])
        else:
            h, w = tensor.shape[-2:]
        item = F.interpolate(tensor[i:i + 1], size=(h, w), mode="bilinear", align_corners=True)
        valid = torch.ones_like(item[:, :1])
        if "random_crop_axis" in params:
            y1, y2, x1, x2 = (int(v[i]) for v in params["random_crop_axis"])
            # RandomCrop pads symmetrically by the shortfall on both sides.
            px, py = max(0, out_w - w), max(0, out_h - h)
            item = F.pad(item, (px, px, py, py))
            valid = F.pad(valid, (px, px, py, py))
            item, valid = item[:, :, y1:y2, x1:x2], valid[:, :, y1:y2, x1:x2]
        if "RandomHorizontalFlip" in params and bool(params["RandomHorizontalFlip"][i]):
            item, valid = item.flip(-1), valid.flip(-1)
        if item.shape[-2:] != (out_h, out_w):
            raise ValueError("Teacher crop geometry does not match the student image.")
        aligned.append(item)
        validity.append(valid.bool())
    return torch.cat(aligned), torch.cat(validity)


def feature_exclusions(mix_label, pixel_valid, feature_size, ignore_label=255, halo=1):
    """Any pasted or padded pixel excludes its cell; expand by a small halo.

    This handles direct alignment conflicts, not the entire receptive field of
    the CNN. Adjacent visible CNN tokens can still contain contextual information
    from the masked image area, as in MRM-style post-encoder latent masking.
    """
    invalid = (mix_label != ignore_label).unsqueeze(1) | ~pixel_valid.bool()
    excluded = F.adaptive_max_pool2d(invalid.float(), feature_size)
    if halo:
        excluded = F.max_pool2d(excluded, 2 * halo + 1, stride=1, padding=halo)
    return excluded.bool()


@torch.no_grad()
def update_ema(teacher, student, decay):
    for target, online in zip(teacher.parameters(), student.parameters()):
        target.mul_(decay).add_(online.detach(), alpha=1.0 - decay)
    # Frozen BN is the G2C default. If normal BN is used, update its state too.
    for target, online in zip(teacher.buffers(), student.buffers()):
        if target.is_floating_point():
            target.mul_(decay).add_(online.detach(), alpha=1.0 - decay)
        else:
            target.copy_(online)


def auxiliary_weight(iteration, settings):
    if settings.OBJECTIVE == "none" or iteration < settings.WARMUP_ITERS:
        return 0.0
    ramp = min(1.0, (iteration - settings.WARMUP_ITERS + 1) / max(1, settings.RAMP_ITERS))
    return settings.WEIGHT * ramp


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 1.x compatibility.
        return torch.load(path, map_location="cpu")


def strip_module(state):
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def build_models(config, checkpoint):
    from core.models import build_classifier
    from core.models.feature_extractor import resnet_feature_extractor

    # The common factory downloads ImageNet weights even when MODEL.WEIGHTS is
    # empty. A complete source checkpoint makes that initialization unnecessary.
    backbone = config.MODEL.NAME.split("_", 1)[1]
    encoder = resnet_feature_extractor(
        backbone, aux=True, pretrained_backbone=False, freeze_bn=config.MODEL.FREEZE_BN
    )
    classifier = build_classifier(config)
    encoder.load_state_dict(strip_module(checkpoint["feature_extractor"]), strict=True)
    classifier.load_state_dict(strip_module(checkpoint["classifier"]), strict=True)
    classifier.requires_grad_(False)
    classifier.eval()
    teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
    rebuilder = SpatialRebuilder(classifier.conv2d_list[0].in_channels,
                                config.SRMP.HIDDEN_DIM, config.SRMP.DEPTH)
    return encoder, classifier, teacher, rebuilder


def seed_worker(worker_id):
    import numpy as np

    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_target_loader(config, workers, generator):
    from core.datasets import build_dataset

    target = build_dataset(config, mode="train", is_source=False, rsc="CTR_O.p")
    if len(target) < config.SOLVER.BATCH_SIZE:
        raise ValueError("Target sampling produced fewer images than one full batch; check CTR_O.p.")
    loader = torch.utils.data.DataLoader(
        target, batch_size=config.SOLVER.BATCH_SIZE, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=generator,
    )
    return target, loader


def atomic_save(state, path):
    temporary = str(path) + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def rng_state(generator):
    import numpy as np

    n = np.random.get_state()
    return {
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(), "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    import numpy as np

    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    generator.set_state(state["loader"])


def validate_config(config, args):
    from srmp_diagnostics import validate_diagnostic_config

    validate_diagnostic_config(config)
    s = config.SRMP
    if s.OBJECTIVE not in ("relation", "probability", "masked_ce", "none"):
        raise ValueError("SRMP.OBJECTIVE must be relation, probability, masked_ce, or none.")
    if s.MAIN_LOSS not in ("reliability", "ce", "snd"):
        raise ValueError("SRMP.MAIN_LOSS must be reliability, ce, or snd.")
    if not 0 < s.MASK_RATIO < 1 or s.TEMPERATURE <= 0:
        raise ValueError("Require 0 < SRMP.MASK_RATIO < 1 and SRMP.TEMPERATURE > 0.")
    if min(s.BLOCK_SIZE, s.HIDDEN_DIM, s.DEPTH) < 1:
        raise ValueError("Block size, hidden dimension, and depth must be positive.")
    if min(s.WEIGHT, s.WARMUP_ITERS, s.RAMP_ITERS, s.EXCLUDE_HALO, s.GRAD_CLIP) < 0:
        raise ValueError("Weights, warmup/ramp, halo, and gradient clip cannot be negative.")
    if s.PREDICTOR_LR_MULT <= 0 or args.workers < 0:
        raise ValueError("Predictor LR multiplier must be positive; workers cannot be negative.")
    if min(config.SOLVER.MAX_ITER, config.SOLVER.STOP_ITER, config.SOLVER.BATCH_SIZE,
           config.SOLVER.CHECKPOINT_PERIOD, config.DTU.FIX_ITERATION) < 1:
        raise ValueError("Iteration counts, batch size, checkpoint and EMA periods must be positive.")
    if not 0 <= config.DTU.EMA_WEIGHT < 1:
        raise ValueError("DTU.EMA_WEIGHT must lie in [0, 1).")
    if config.DTU.DYNAMIC:
        raise ValueError("SRMP uses fixed EMA; pass DTU.DYNAMIC False, as in the G2C ReMask task.")
    if config.MODEL.NAME != "deeplab_resnet101" or config.MODEL.NUM_CLASSES != 19:
        raise ValueError("This first SRMP implementation targets 19-class DeepLabV2-ResNet101.")
    if config.DATASETS.TARGET_TRAIN != "cityscapes_train":
        raise ValueError("Use DATASETS.TARGET_TRAIN cityscapes_train for this G2C trainer.")
    if not 0 <= config.TCR.TOPK_CANDIDATE <= 1:
        raise ValueError("TCR.TOPK_CANDIDATE must lie in [0, 1].")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("SRMP currently supports one GPU per job, as in the reference SLURM.")


def train(config, args):
    from core.apis.inference import run_candidate, run_test
    from core.solver import adjust_learning_rate
    from core.utils.losses import pseudo_labels_probs, update_running_conf
    from core.utils.metric_logger import MetricLogger
    from datasets.generate_city_label_info import gen_lb_info
    from train_ReMask import (
        _apply_loss_weight_floor, _compute_reliability, _ensure_remask_candidates,
        _update_mixup_sampling,
    )

    logger = logging.getLogger("DTST.trainer")
    device = torch.device(config.MODEL.DEVICE)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The inherited G2C candidate/evaluation framework requires CUDA; use --self-test on CPU.")
    torch.cuda.set_device(device.index if device.index is not None else args.local_rank)
    checkpoint = load_checkpoint(args.resume_training or config.resume)
    if not isinstance(checkpoint, dict) or not {"feature_extractor", "classifier"} <= checkpoint.keys():
        raise ValueError("Checkpoint must contain feature_extractor and classifier state dictionaries.")
    if args.resume_training:
        if checkpoint.get("srmp_format") != 1:
            raise ValueError("--resume-training requires an SRMP srmp_last.pth checkpoint.")
        if checkpoint["settings"] != dict(config.SRMP):
            raise ValueError("Resume with the same SRMP settings saved in the checkpoint.")
    elif "srmp_format" in checkpoint:
        raise ValueError("Use --resume-training to continue SRMP; resume is for the source initialization.")

    encoder, classifier, teacher, rebuilder = build_models(config, checkpoint)
    directions, gram = source_kernel_geometry(classifier)
    logger.info("Frozen source kernels: %s; Gram eigenvalue range: %.6f .. %.6f",
                tuple(directions.shape), torch.linalg.eigvalsh(gram).min(), torch.linalg.eigvalsh(gram).max())
    del directions
    if config.SRMP.OBJECTIVE == "probability":
        gram = torch.eye(config.MODEL.NUM_CLASSES)
    encoder, classifier = encoder.to(device), classifier.to(device)
    teacher, rebuilder = teacher.to(device), rebuilder.to(device)
    gram = gram.to(device)
    optimizer = torch.optim.SGD([
        {"params": encoder.parameters(), "lr_mult": 1.0},
        {"params": rebuilder.parameters(), "lr_mult": config.SRMP.PREDICTOR_LR_MULT},
    ], lr=config.SOLVER.BASE_LR, momentum=config.SOLVER.MOMENTUM,
        weight_decay=config.SOLVER.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    generator = torch.Generator().manual_seed(args.seed)
    running_conf = torch.full((config.MODEL.NUM_CLASSES,), 0.001, device=device)
    uncertainty = {name: None for name in ("entropy_mean", "entropy_var", "kl_mean", "kl_var")}
    iteration, best_miou = 0, -1.0
    if args.resume_training:
        encoder.load_state_dict(checkpoint["student_feature_extractor"], strict=True)
        rebuilder.load_state_dict(checkpoint["srmp_rebuilder"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        running_conf = checkpoint["running_conf"].to(device)
        uncertainty = {k: v.to(device) if v is not None else None
                       for k, v in checkpoint["uncertainty"].items()}
        iteration, best_miou = int(checkpoint["iteration"]), float(checkpoint["best_miou"])
        if not torch.allclose(gram.cpu(), checkpoint["source_gram"].cpu(), atol=1e-6):
            raise ValueError("Saved source geometry does not match the frozen classifier.")
    endpoint = min(config.SOLVER.MAX_ITER, config.SOLVER.STOP_ITER)
    if iteration >= endpoint:
        raise ValueError("The requested endpoint has already been reached by this checkpoint.")
    _ensure_remask_candidates(config, teacher, classifier, 0, False)
    if (args.resume_training and config.TCR.OPEN and config.TCR.UPDATE_FREQUENCY > 0
            and iteration % config.TCR.UPDATE_FREQUENCY == 0):
        # A checkpoint is written before the candidate refresh. Reconstruct the
        # current candidates if the previous job stopped at that boundary.
        run_candidate(config, teacher, classifier, 0, False)
        gen_lb_info(config, "CTR")
        gen_lb_info(config, "CTR_O")
    target, loader = build_target_loader(config, args.workers, generator)
    if args.resume_training:
        restore_rng(checkpoint["rng"], generator)
    del checkpoint
    _update_mixup_sampling(config, target, running_conf)
    encoder.train()
    rebuilder.train()
    teacher.eval()
    classifier.eval()
    meters = MetricLogger(delimiter="  ")
    started, start_iteration = time.time(), iteration
    logger.info("SRMP: objective=%s, main=%s, target samples=%d, iterations=%d..%d",
                config.SRMP.OBJECTIVE, config.SRMP.MAIN_LOSS, len(target), iteration, endpoint)
    logger.info("Classifier frozen; no prototype contrastive loss or DTU meta optimization.")

    diagnostics = None
    if config.SRMP_DIAG.ENABLED:
        from srmp_diagnostics import TrainingDiagnostics

        diagnostics = TrainingDiagnostics(config, device)
        if config.SRMP_DIAG.INITIAL:
            diagnostics.run(iteration, encoder, classifier, teacher, rebuilder, gram, force=True)

    while iteration < endpoint:
        for images, _unused_target_gt, _names, params, full_images, mix_label in loader:
            step_start = time.time()
            images = images.to(device, non_blocking=True)
            full_images = full_images.to(device, non_blocking=True)
            mix_label = mix_label.to(device, non_blocking=True).long()
            image_size = images.shape[-2:]
            lr = adjust_learning_rate(config.SOLVER.LR_METHOD, config.SOLVER.BASE_LR,
                                      iteration, config.SOLVER.MAX_ITER, config.SOLVER.LR_POWER)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["lr_mult"]
            optimizer.zero_grad(set_to_none=True)
            weight_rel = auxiliary_weight(iteration, config.SRMP)

            # Same full-image teacher -> crop pseudo-label path as ReMask.
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
                full_features = teacher(full_images)[1]
                full_logits = classifier(full_features, full_images.shape[-2:])
                full_prob = full_logits.float().softmax(dim=1)
                running_conf = update_running_conf(full_prob, running_conf, 0.001)
                aligned_logits, pixel_valid = align_full_image(full_logits, params, image_size)
                teacher_prob = aligned_logits.float().softmax(dim=1)
                del full_features, full_logits, full_prob, aligned_logits

            with torch.cuda.amp.autocast(enabled=args.amp):
                features = encoder(images)[1]
                logits = classifier(features, image_size)
            with torch.no_grad():
                pseudo = teacher_prob.argmax(dim=1)
                if config.SRMP.MAIN_LOSS == "snd":
                    pseudo, _, _ = pseudo_labels_probs(teacher_prob, running_conf, 0.001)
                reliability = torch.ones_like(teacher_prob[:, :1])
                if config.SRMP.MAIN_LOSS == "reliability":
                    reliability = _compute_reliability(
                        config, teacher_prob, logits.detach().float().softmax(dim=1), uncertainty
                    )
                if mix_label.shape[-2:] != image_size:
                    mix_label = F.interpolate(mix_label[:, None].float(), image_size,
                                              mode="nearest")[:, 0].long()
                pasted = mix_label != config.INPUT.IGNORE_LABEL
                pseudo = torch.where(pixel_valid[:, 0], pseudo, config.INPUT.IGNORE_LABEL)
                pseudo = torch.where(pasted, mix_label, pseudo)
                reliability = torch.where(pasted[:, None], torch.ones_like(reliability), reliability)
                pl_weight = _apply_loss_weight_floor(config, reliability).squeeze(1)
                pl_weight *= (pseudo != config.INPUT.IGNORE_LABEL)
            ce_map = F.cross_entropy(logits.float(), pseudo.long(),
                                     ignore_index=config.INPUT.IGNORE_LABEL, reduction="none")
            loss_pl = (ce_map * pl_weight).sum() / pl_weight.sum().clamp_min(1e-6)
            loss_aux = loss_pl.new_zeros(())
            mask_fraction = selected_fraction = 0.0

            if weight_rel > 0:
                # Additional clean teacher pass at the student's crop geometry.
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
                    clean_crop, _ = align_full_image(full_images, params, image_size)
                    clean_features = teacher(clean_crop)[1]
                    clean_logits = classifier(clean_features)
                    q = (clean_logits.float() / config.SRMP.TEMPERATURE).softmax(dim=1)
                    del clean_crop, clean_features, clean_logits
                if q.shape[-2:] != features.shape[-2:]:
                    raise RuntimeError("Student and clean teacher feature grids are misaligned.")
                hidden = sample_block_mask(features.shape[0], *features.shape[-2:],
                                           config.SRMP.MASK_RATIO, config.SRMP.BLOCK_SIZE, device)
                excluded = feature_exclusions(mix_label, pixel_valid, features.shape[-2:],
                                               config.INPUT.IGNORE_LABEL, config.SRMP.EXCLUDE_HALO)
                selected = (hidden & ~excluded)[:, 0]
                # Samples without any visible context cannot provide a contextual target.
                has_context = (~(hidden | excluded)).flatten(1).any(dim=1)
                selected = selected & has_context[:, None, None]
                with torch.cuda.amp.autocast(enabled=args.amp):
                    rebuilt = rebuilder(features, hidden | excluded)
                    predicted_logits = classifier(rebuilt)
                if config.SRMP.OBJECTIVE == "masked_ce":
                    aux_map = F.cross_entropy(predicted_logits.float(), q.argmax(dim=1), reduction="none")
                else:
                    p = (predicted_logits.float() / config.SRMP.TEMPERATURE).softmax(dim=1)
                    # Keep the class-space metric in FP32 even when AMP is enabled.
                    aux_map = relational_cosine_map(p, q, gram)
                loss_aux = masked_mean(aux_map, selected)
                mask_fraction = hidden.float().mean().item()
                selected_fraction = selected.float().mean().item()

            total_loss = loss_pl + weight_rel * loss_aux
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite SRMP loss at iteration {}".format(iteration + 1))
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if config.SRMP.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(rebuilder.parameters()),
                                               config.SRMP.GRAD_CLIP)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
            iteration += 1
            if optimizer_stepped and iteration % config.DTU.FIX_ITERATION == 0:
                update_ema(teacher, encoder, config.DTU.EMA_WEIGHT)
            _update_mixup_sampling(config, target, running_conf)
            meters.update(loss_seg=loss_pl.item(), loss_rel=loss_aux.item(),
                          total_loss=total_loss.item(), lambda_rel=weight_rel,
                          mask_rate=mask_fraction, selected_rate=selected_fraction,
                          reliability=reliability.mean().item(), time=time.time() - step_start)
            if iteration % (1 if endpoint <= 100 else 20) == 0 or iteration == endpoint:
                eta = meters.time.global_avg * (endpoint - iteration)
                logger.info("eta: %s  iter: %d  %s  lr: %.6f  max mem: %.0f",
                            str(datetime.timedelta(seconds=int(eta))), iteration, meters, lr,
                            torch.cuda.max_memory_allocated() / (1024 ** 2))

            # Release crop-sized activations before full-resolution validation
            # or the next teacher pass (particularly useful on MIG GPUs).
            del images, full_images, features, logits, teacher_prob, ce_map, total_loss, loss_pl, loss_aux
            if weight_rel > 0:
                del rebuilt, predicted_logits, q, aux_map
                if config.SRMP.OBJECTIVE != "masked_ce":
                    del p

            if iteration % config.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == endpoint:
                inference_state = {"iteration": iteration, "feature_extractor": teacher.state_dict(),
                                   "classifier": classifier.state_dict(), "method": "SRMP",
                                   "settings": dict(config.SRMP)}
                output = Path(config.OUTPUT_DIR)
                atomic_save(inference_state, output / "srmp_model_iter{:06d}.pth".format(iteration))
                if not args.skip_test:
                    miou = float(run_test(config, teacher, classifier, 0, False))
                    if miou > best_miou:
                        best_miou = miou
                        atomic_save(inference_state, output / "srmp_best_val.pth")
                atomic_save({
                    **inference_state, "srmp_format": 1,
                    "student_feature_extractor": encoder.state_dict(),
                    "srmp_rebuilder": rebuilder.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "source_gram": gram,
                    "running_conf": running_conf, "uncertainty": uncertainty,
                    "best_miou": best_miou, "rng": rng_state(generator),
                }, output / "srmp_last.pth")
            if diagnostics is not None:
                diagnostics.run(iteration, encoder, classifier, teacher, rebuilder, gram,
                                force=(iteration == endpoint))
            if iteration >= endpoint:
                break
            if config.TCR.OPEN and config.TCR.UPDATE_FREQUENCY > 0 and iteration % config.TCR.UPDATE_FREQUENCY == 0:
                run_candidate(config, teacher, classifier, 0, False)
                gen_lb_info(config, "CTR")
                gen_lb_info(config, "CTR_O")
                target, loader = build_target_loader(config, args.workers, generator)
                _update_mixup_sampling(config, target, running_conf)
                break

    logger.info("Finished %d updates in %s; final checkpoint: %s",
                iteration - start_iteration, str(datetime.timedelta(seconds=int(time.time() - started))),
                Path(config.OUTPUT_DIR) / "srmp_last.pth")


def self_test():
    """Small CPU checks independent of datasets and the training dependencies."""
    torch.manual_seed(17)

    class TinyASPP(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2d_list = nn.ModuleList([
                nn.Conv2d(8, 3, 3, padding=d, dilation=d) for d in (1, 2)
            ])

        def forward(self, x):
            return sum(branch(x) for branch in self.conv2d_list)

    head = TinyASPP().requires_grad_(False)
    directions, gram = source_kernel_geometry(head)
    assert directions.shape == (3, 8 * 17), "Shared ASPP centers must merge."
    assert torch.allclose(gram.diag(), torch.ones(3), atol=1e-6)
    assert torch.linalg.eigvalsh(gram).min() > -1e-6
    a = torch.randn(2, 3, 9, 13, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    p, q = a.softmax(1), b.softmax(1)
    exact = 1 - F.cosine_similarity(torch.einsum("bchw,cd->bdhw", p, directions),
                                    torch.einsum("bchw,cd->bdhw", q.detach(), directions), dim=1)
    actual = relational_cosine_map(p, q, gram)
    assert torch.allclose(actual, exact, atol=2e-6), "Gram cosine must equal explicit p @ W."
    grad_actual = torch.autograd.grad(actual.mean(), a, retain_graph=True)[0]
    grad_exact = torch.autograd.grad(exact.mean(), a, retain_graph=True)[0]
    assert torch.allclose(grad_actual, grad_exact, atol=1e-6)
    actual.mean().backward()
    assert b.grad is None, "Teacher probabilities must be stop-gradient."
    assert relational_cosine_map(q, q, gram).abs().max() < 1e-6
    naive = 1 - F.cosine_similarity(torch.einsum("cd,bdhw->bchw", gram, p.detach()),
                                    torch.einsum("cd,bdhw->bchw", gram, q.detach()), dim=1)
    assert not torch.allclose(actual.detach(), naive, atol=1e-6), "A @ p is a different metric."

    feature = torch.randn(2, 8, 9, 13, requires_grad=True)
    hidden = sample_block_mask(2, 9, 13, 0.5, 4, "cpu")
    assert hidden.any() and (~hidden).any() and hidden.shape == (2, 1, 9, 13)
    predictor = SpatialRebuilder(8, 16, 2)
    rebuilt = predictor(feature, hidden)
    changed = torch.where(hidden, feature + 1000, feature)
    assert torch.allclose(rebuilt, predictor(changed, hidden)), "Hidden features must not leak into predictor."
    prediction = head(rebuilt).softmax(dim=1)
    loss = masked_mean(relational_cosine_map(prediction, q.detach(), gram), hidden[:, 0])
    loss.backward()
    assert feature.grad.abs().sum() > 0, "Auxiliary loss must train the encoder through visible context."
    assert feature.grad.masked_select(hidden.expand_as(feature)).abs().sum() == 0
    assert predictor.mask_token.grad.abs().sum() > 0
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in predictor.parameters())
    assert all(v.grad is None for v in head.parameters()), "Frozen classifier must not receive gradients."
    empty_logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    empty = masked_mean(empty_logits[:, 0], torch.zeros(1, 2, 2, dtype=torch.bool))
    empty.backward()
    assert empty.item() == 0 and torch.count_nonzero(empty_logits.grad) == 0

    grid = torch.arange(32.0).view(1, 1, 4, 8)
    params = {"rescale_size": [torch.tensor([4]), torch.tensor([8])],
              "random_crop_axis": [torch.tensor([1]), torch.tensor([4]), torch.tensor([2]), torch.tensor([7])],
              "RandomHorizontalFlip": torch.tensor([True])}
    crop, valid = align_full_image(grid, params, (3, 5))
    assert torch.equal(crop, grid[:, :, 1:4, 2:7].flip(-1)) and valid.all()
    resize, valid = align_full_image(grid, {"resize": [torch.tensor([2]), torch.tensor([4])]}, (2, 4))
    assert resize.shape[-2:] == (2, 4) and valid.all()
    params = {"rescale_size": [torch.tensor([4]), torch.tensor([8])],
              "random_crop_axis": [torch.tensor([0]), torch.tensor([6]), torch.tensor([0]), torch.tensor([8])]}
    padded, valid = align_full_image(grid, params, (6, 8))
    assert not valid[:, :, :2].any() and torch.equal(padded[:, :, 2:], grid)
    mix = torch.full((1, 6, 8), 255)
    mix[:, 4, 4] = 1
    excluded = feature_exclusions(mix, valid, (3, 4), halo=0)
    assert excluded[0, 0, 0].all() and excluded[0, 0, 2, 2]

    online = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
    teacher = copy.deepcopy(online).requires_grad_(False)
    old = teacher[0].weight.detach().clone()
    with torch.no_grad():
        online[0].weight.add_(2)
        online[1].running_mean.add_(4)
        online[1].num_batches_tracked.add_(1)
    update_ema(teacher, online, 0.75)
    assert torch.allclose(teacher[0].weight, old + 0.5)
    assert torch.allclose(teacher[1].running_mean, torch.ones(2))
    assert teacher[1].num_batches_tracked.item() == 1
    print("SRMP self-test PASSED: ASPP geometry, exact loss/gradient, stop-gradient, masking,")
    print("context gradients, frozen classifier, empty masks, crop/flip/padding, exclusions, EMA.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-cfg", "--config-file", default="configs/deeplabv2_r101_SND_G2C.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Optional CUDA mixed precision.")
    parser.add_argument("--skip-test", action="store_true", help="Skip validation mIoU; still save checkpoints and run enabled SRMP_DIAG audits.")
    parser.add_argument("--temp_train", action="store_true", help="10 real iterations, including the auxiliary branch.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic mathematical and gradient checks on CPU.")
    parser.add_argument("--resume-training", default="", help="Continue from srmp_last.pth, including EMA and optimizer.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="KEY VALUE config overrides, as in the existing SLURM.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    import numpy as np
    from core.configs import cfg
    from core.utils.logger import setup_logger

    config = cfg.clone()
    config.defrost()
    add_srmp_config(config)
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.resume = config.resume or "pretrained/G2C_model_iter020000.pth"
    if config.OUTPUT_DIR in ("", "."):
        config.OUTPUT_DIR = "results_gta5_to_cityscapes/srmp"
    if args.temp_train:
        config.SOLVER.MAX_ITER = config.SOLVER.STOP_ITER = 10
        config.SOLVER.CHECKPOINT_PERIOD = 10
        config.SRMP.WARMUP_ITERS, config.SRMP.RAMP_ITERS = 0, 0
    validate_config(config, args)
    if not os.path.isfile(args.resume_training or config.resume):
        raise FileNotFoundError(args.resume_training or config.resume)
    config.freeze()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = not (os.name == "nt" or args.temp_train)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("DTST", config.OUTPUT_DIR, 0)
    logger.info("SRMP config:\n%s", config)
    with open(Path(config.OUTPUT_DIR) / "srmp_config.yaml", "w", encoding="utf-8") as handle:
        handle.write(config.dump())
    with open(Path(config.OUTPUT_DIR) / "srmp_arguments.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    train(config, args)


if __name__ == "__main__":
    main()
