#!/usr/bin/env python3
"""Export aligned, post-hoc visualization assets for the two-stage ReMask model.

The exporter is deliberately read-only with respect to training results.  It
selects high-per-image-mIoU SUIM training samples using the final Stage-2
checkpoint, then reconstructs the diagnostic maps from the saved Stage-1 and
Stage-2 states.  SUIM-train is used because the Stage-2 temporal snapshots are
stored only for target images seen during adaptation.

Two UWRG probes are written:

* ``uwrg_probe_method.png`` uses the exact perturbation strength of the run.
* ``uwrg_probe_display.png`` uses a stronger, explicitly labelled diagnostic
  strength so attenuation/backscatter is easy to see in a paper figure.

The stronger display probe must not be presented as the training setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.configs.defaults import _C as DEFAULT_CFG
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.models import build_classifier
from core.models.feature_extractor import resnet_feature_extractor
from core.utils.misc import strip_prefix_if_present
from train_ReMask_special import (
    _compute_class_balance_reliability,
    _compute_temporal_label_raw,
    _compute_underwater_stability,
    _ensure_classifier_buffers_from_state,
    _image_space,
    _normalized_space,
    _read_temporal_label_snapshot,
    _underwater_reliability_perturb,
)


CLASS_NAMES = [
    "Background",
    "Diver",
    "Plant",
    "Wreck",
    "Robot",
    "Reef",
    "Fish",
    "Sea floor",
]

SUIM_PALETTE = np.asarray(
    [
        (0, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)

MAP_SPECS = {
    "entropy": (0.0, 1.0, "gray"),
    "kl_normalized": (0.0, 1.0, "gray"),
    "entropy_kl_reliability": (0.0, 1.0, "gray"),
    "uwrg_reliability_method": (0.0, 1.0, "gray"),
    "uwrg_reliability_display": (0.0, 1.0, "gray"),
    "uwrg_downweight_method": (0.0, None, "gray"),
    "uwrg_downweight_display": (0.0, None, "gray"),
    "temporal_agreement": (0.0, 1.0, "gray"),
    "temporal_reliability": (0.0, 1.0, "gray"),
    "class_balance": (None, None, "gray"),
    "temporal_class_weight": (None, None, "gray"),
    "final_weight_with_uwrg": (None, None, "gray"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export high-mIoU aligned visualization assets for ReMask."
    )
    parser.add_argument(
        "--config-file",
        default="configs/deeplabv2_r101_SND_SUIM.yaml",
    )
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--dataset", default="suim_train")
    parser.add_argument("--setting-name", required=True)
    parser.add_argument(
        "--output-dir",
        default="validate_results/remask_visualization_assets",
    )
    parser.add_argument("--num-select", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument(
        "--selection-mode",
        choices=("top_miou", "balanced"),
        default="balanced",
        help=(
            "top_miou selects the highest per-image mIoU after the error filter; "
            "balanced selects high-mIoU samples whose diagnostic maps are clearer."
        ),
    )
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.75,
        help="Quality-rank weight used by balanced selection.",
    )
    parser.add_argument(
        "--min-error-fraction",
        type=float,
        default=0.002,
        help="Avoid visually identical GT/prediction pairs.",
    )
    parser.add_argument(
        "--max-error-fraction",
        type=float,
        default=0.25,
        help="Keep predictions close enough to GT for a high-quality example.",
    )
    parser.add_argument(
        "--min-present-classes",
        type=int,
        default=3,
        help="Reject trivial images with too few GT classes (including Background).",
    )
    parser.add_argument(
        "--min-foreground-classes",
        type=int,
        default=2,
        help="Require at least this many non-Background GT classes.",
    )
    parser.add_argument("--method-probe-strength", type=float, required=True)
    parser.add_argument(
        "--display-probe-strength",
        type=float,
        default=0.65,
        help="Diagnostic-only strength for a visibly different probe image.",
    )
    parser.add_argument("--uwrg-floor", type=float, required=True)
    parser.add_argument("--uwrg-lambda", type=float, default=1.0)
    parser.add_argument("--uwrg-mode", choices=("pixel", "segment"), default="segment")
    parser.add_argument("--segment-stride", type=int, default=4)
    parser.add_argument("--min-segment-pixels", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--kl-scale", type=float, default=1.0)
    parser.add_argument("--temporal-floor", type=float, default=0.5)
    parser.add_argument("--class-balance-gamma", type=float, default=0.25)
    parser.add_argument("--class-balance-min", type=float, default=0.7)
    parser.add_argument("--class-balance-max", type=float, default=1.3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--map-rendering",
        choices=("contrast", "absolute"),
        default="contrast",
        help=(
            "Rendering of the main grayscale PNGs. contrast uses per-image "
            "1st/99th percentiles; *_absolute.png always preserves fixed limits."
        ),
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save lossless float arrays for quantitative reuse.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_cfg(args: argparse.Namespace) -> Any:
    cfg = DEFAULT_CFG.clone()
    cfg.merge_from_file(str(resolve_path(args.config_file)))
    cfg.defrost()
    cfg.DATASETS.TARGET_TRAIN = args.dataset
    cfg.DATASETS.TEST = args.dataset
    cfg.INPUT.INPUT_SIZE_TEST = (args.width, args.height)
    cfg.REMASK.ALPHA = args.alpha
    cfg.REMASK.BETA = args.beta
    cfg.REMASK.TAU = args.tau
    cfg.REMASK.KL_SCALE = args.kl_scale
    cfg.REMASK.TEMPORAL_FLOOR = args.temporal_floor
    cfg.REMASK.CLASS_BALANCE_GAMMA = args.class_balance_gamma
    cfg.REMASK.CLASS_BALANCE_MIN = args.class_balance_min
    cfg.REMASK.CLASS_BALANCE_MAX = args.class_balance_max
    cfg.REMASK.USE_CLASS_BALANCE = True
    cfg.REMASK.HISTORY_USE_CONF = True
    cfg.freeze()
    return cfg


def load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def build_feature_extractor(cfg: Any) -> torch.nn.Module:
    _model_name, backbone_name = cfg.MODEL.NAME.split("_")
    if not backbone_name.startswith("resnet"):
        raise NotImplementedError("Only the ReMask ResNet checkpoints are supported.")
    return resnet_feature_extractor(
        backbone_name,
        pretrained_weights=None,
        aux=True,
        pretrained_backbone=False,
        freeze_bn=cfg.MODEL.FREEZE_BN,
    )


def checkpoint_weights(
    checkpoint: Dict[str, Any], role: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if role == "teacher":
        feature_keys = ("teacher_feature_extractor", "feature_extractor")
        classifier_keys = ("teacher_classifier", "classifier")
    elif role == "student":
        feature_keys = ("student_feature_extractor", "feature_extractor")
        classifier_keys = ("student_classifier", "classifier")
    else:
        raise ValueError("Unknown role: {}".format(role))

    feature_key = next((key for key in feature_keys if key in checkpoint), None)
    classifier_key = next((key for key in classifier_keys if key in checkpoint), None)
    if feature_key is None or classifier_key is None:
        raise KeyError(
            "Checkpoint lacks {} weights (feature={}, classifier={}).".format(
                role, feature_keys, classifier_keys
            )
        )
    return checkpoint[feature_key], checkpoint[classifier_key]


def build_checkpoint_model(
    cfg: Any, checkpoint: Dict[str, Any], role: str, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    feature_extractor = build_feature_extractor(cfg)
    classifier = build_classifier(cfg)
    feature_weights, classifier_weights = checkpoint_weights(checkpoint, role)
    feature_extractor.load_state_dict(
        strip_prefix_if_present(feature_weights, "module."), strict=True
    )
    # Match the training/continuation loader: Stage-1 student checkpoints may
    # contain prototype-memory buffers that are attached lazily during train.
    # Register those buffers before strict loading so no checkpoint state is
    # discarded and every other architecture mismatch remains fatal.
    _ensure_classifier_buffers_from_state(classifier, classifier_weights)
    classifier_state = strip_prefix_if_present(classifier_weights, "module.")
    classifier.load_state_dict(classifier_state, strict=True)
    feature_extractor.to(device).eval()
    classifier.to(device).eval()
    return feature_extractor, classifier


@torch.no_grad()
def forward_prob(
    feature_extractor: torch.nn.Module,
    classifier: torch.nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    _, features = feature_extractor(image)
    logits = classifier(features, image.shape[-2:])
    return F.softmax(logits, dim=1)


def make_dataset(cfg: Any, dataset_name: str) -> Any:
    # ``mode='posthoc'`` intentionally chooses the simple SuimDataSet even when
    # the catalog name ends in _train.  No augmentation is needed here.
    return DatasetCatalog.get(
        dataset_name,
        mode="posthoc",
        num_classes=cfg.MODEL.NUM_CLASSES,
        max_iters=None,
        transform=None,
        rsc=None,
        cfg=cfg,
    )


def image_to_tensor(image: Image.Image, width: int, height: int) -> torch.Tensor:
    image = image.convert("RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def label_to_tensor(label: Image.Image, width: int, height: int) -> torch.Tensor:
    resized = label.resize((width, height), Image.NEAREST)
    return torch.from_numpy(np.asarray(resized, dtype=np.int64)).unsqueeze(0)


def get_sample(dataset: Any, index: int, args: argparse.Namespace) -> Dict[str, Any]:
    image, label, name = dataset[index]
    return {
        "index": index,
        "name": str(name),
        "stem": safe_stem(name),
        "image": image_to_tensor(image, args.width, args.height),
        "label": label_to_tensor(label, args.width, args.height),
    }


def safe_stem(name: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(name)).stem) or "sample"


def normalized_history_key(name: Any) -> str:
    return Path(str(name)).stem.lower()


def history_index(history: Dict[str, Any]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for key in history:
        normalized = normalized_history_key(key)
        if normalized in output and output[normalized] != key:
            raise RuntimeError("Ambiguous temporal-history key: {}".format(normalized))
        output[normalized] = key
    return output


def resize_history(
    hist_item: Any, size: Tuple[int, int], device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    hist_label, hist_conf = _read_temporal_label_snapshot(hist_item)
    hist_label_t = torch.as_tensor(hist_label, device=device).float()
    while hist_label_t.ndim < 4:
        hist_label_t = hist_label_t.unsqueeze(0)
    hist_label_t = F.interpolate(hist_label_t, size=size, mode="nearest").long()

    hist_conf_t: Optional[torch.Tensor]
    if hist_conf is None:
        hist_conf_t = None
    else:
        hist_conf_t = torch.as_tensor(hist_conf, device=device).float()
        while hist_conf_t.ndim < 4:
            hist_conf_t = hist_conf_t.unsqueeze(0)
        hist_conf_t = F.interpolate(
            hist_conf_t, size=size, mode="bilinear", align_corners=True
        )
    return hist_label_t, hist_conf_t


def sample_miou(
    prediction: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    ignore_label: int,
) -> Tuple[float, List[float]]:
    prediction_np = prediction.detach().cpu().numpy().astype(np.int64)
    label_np = label.detach().cpu().numpy().astype(np.int64)
    valid = label_np != ignore_label
    per_class: List[float] = []
    for class_id in range(num_classes):
        pred_mask = (prediction_np == class_id) & valid
        gt_mask = (label_np == class_id) & valid
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union > 0:
            per_class.append(float(np.logical_and(pred_mask, gt_mask).sum() / union))
    return (float(np.mean(per_class)) if per_class else float("nan"), per_class)


def error_fraction(
    prediction: torch.Tensor, label: torch.Tensor, ignore_label: int
) -> float:
    valid = label != ignore_label
    if not bool(valid.any()):
        return float("nan")
    return float(((prediction != label) & valid).sum().item() / valid.sum().item())


def present_classes(label: torch.Tensor, ignore_label: int) -> List[int]:
    values = torch.unique(label[label != ignore_label]).detach().cpu().tolist()
    return sorted(int(value) for value in values)


def scalar_from_state(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.as_tensor(value, device=device).float()


def fixed_entropy_kl_maps(
    cfg: Any,
    teacher_prob: torch.Tensor,
    student_prob: torch.Tensor,
    uncertainty_stats: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Recreate Entropy-KL reliability without mutating checkpoint statistics."""
    eps = float(cfg.REMASK.EPSILON)
    entropy_raw = -torch.sum(
        teacher_prob * torch.log(teacher_prob.clamp_min(1e-8)), dim=1, keepdim=True
    )
    kl_raw = torch.sum(
        teacher_prob
        * (
            torch.log(teacher_prob.clamp_min(1e-8))
            - torch.log(student_prob.clamp_min(1e-8))
        ),
        dim=1,
        keepdim=True,
    )

    def sigmoid_normalize(value: torch.Tensor, prefix: str) -> torch.Tensor:
        mean = scalar_from_state(uncertainty_stats.get(prefix + "_mean"), value.device)
        var = scalar_from_state(uncertainty_stats.get(prefix + "_var"), value.device)
        if mean is None or var is None:
            # A source-only checkpoint has no running stats.  Per-image fallback
            # is recorded in metadata by the caller and is only diagnostic.
            mean = value.mean().detach()
            var = value.var(unbiased=False).detach().clamp_min(eps)
        return torch.sigmoid((value - mean) / torch.sqrt(var + eps))

    entropy_normalized = sigmoid_normalize(entropy_raw, "entropy")
    kl_normalized = sigmoid_normalize(kl_raw, "kl")
    energy = (
        float(cfg.REMASK.ALPHA) * entropy_normalized
        + float(cfg.REMASK.BETA) * kl_normalized * float(cfg.REMASK.KL_SCALE)
    )
    reliability = torch.exp(-energy / float(cfg.REMASK.TAU))
    return {
        "entropy_raw": entropy_raw,
        "entropy": entropy_raw / math.log(cfg.MODEL.NUM_CLASSES),
        "kl_raw": kl_raw,
        "kl_normalized": kl_normalized,
        "entropy_normalized": entropy_normalized,
        "entropy_kl_reliability": reliability,
    }


def seed_probe(
    seed: int, image: torch.Tensor, strength: float, cfg: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probe_image = _underwater_reliability_perturb(
        _image_space(image, cfg), strength
    )
    return probe_image, _normalized_space(probe_image, cfg)


@torch.no_grad()
def uwrg_maps(
    teacher_fea: torch.nn.Module,
    teacher_cls: torch.nn.Module,
    image: torch.Tensor,
    teacher_prob: torch.Tensor,
    pseudo_label: torch.Tensor,
    args: argparse.Namespace,
    cfg: Any,
    strength: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    probe_image, probe_input = seed_probe(seed, image, strength, cfg)
    probe_prob = forward_prob(teacher_fea, teacher_cls, probe_input)
    stability, agreement, _segments = _compute_underwater_stability(
        teacher_prob,
        probe_prob,
        pseudo_label,
        ignore_label=255,
        mode=args.uwrg_mode,
        segment_stride=args.segment_stride,
        min_segment_pixels=args.min_segment_pixels,
    )
    gate = float(args.uwrg_floor) + (1.0 - float(args.uwrg_floor)) * stability
    effective_gate = gate.pow(float(args.uwrg_lambda))
    return {
        "probe": probe_image,
        "probe_prediction": probe_prob.argmax(dim=1),
        "agreement": agreement,
        "stability": stability,
        "reliability": effective_gate,
        "downweight": 1.0 - effective_gate,
    }


def tensor_map(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    while array.ndim > 2:
        array = array[0]
    return array


def tensor_image(value: torch.Tensor) -> np.ndarray:
    return (
        value[0]
        .detach()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def label_rgb(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.int64)
    rgb = np.full((*label.shape, 3), 128, dtype=np.uint8)
    valid = (label >= 0) & (label < len(SUIM_PALETTE))
    rgb[valid] = SUIM_PALETTE[label[valid]]
    return rgb


def percentile_rank(values: Sequence[float], higher_is_better: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not higher_is_better:
        array = -array
    order = np.argsort(np.argsort(array))
    if len(array) <= 1:
        return np.ones_like(array)
    return order.astype(np.float64) / float(len(array) - 1)


def atomic_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def map_stats(value: torch.Tensor) -> Dict[str, float]:
    array = tensor_map(value)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p01": float(np.percentile(array, 1)),
        "p50": float(np.percentile(array, 50)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def compute_maps(
    sample: Dict[str, Any],
    cfg: Any,
    args: argparse.Namespace,
    device: torch.device,
    stage1_models: Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module],
    stage2_models: Tuple[torch.nn.Module, torch.nn.Module],
    stage1_state: Dict[str, Any],
    stage2_state: Dict[str, Any],
    temporal_history: Dict[str, Any],
    temporal_keys: Dict[str, str],
) -> Dict[str, Any]:
    stage1_teacher_fea, stage1_teacher_cls, stage1_student_fea, stage1_student_cls = (
        stage1_models
    )
    stage2_teacher_fea, stage2_teacher_cls = stage2_models
    image = sample["image"].to(device)
    label = sample["label"].to(device)

    with torch.no_grad():
        stage1_teacher_prob = forward_prob(stage1_teacher_fea, stage1_teacher_cls, image)
        stage1_student_prob = forward_prob(stage1_student_fea, stage1_student_cls, image)
        stage2_prob = forward_prob(stage2_teacher_fea, stage2_teacher_cls, image)
    stage1_prediction = stage1_teacher_prob.argmax(dim=1)
    stage2_prediction = stage2_prob.argmax(dim=1)

    stage1_training_state = stage1_state.get("training_state", {})
    uncertainty_stats = stage1_training_state.get("uncertainty_stats", {})
    entropy_kl = fixed_entropy_kl_maps(
        cfg, stage1_teacher_prob, stage1_student_prob, uncertainty_stats
    )

    probe_seed = args.seed + int(sample["index"]) * 7919
    stage1_uwrg_method = uwrg_maps(
        stage1_teacher_fea,
        stage1_teacher_cls,
        image,
        stage1_teacher_prob,
        stage1_prediction,
        args,
        cfg,
        args.method_probe_strength,
        probe_seed,
    )
    stage1_uwrg_display = uwrg_maps(
        stage1_teacher_fea,
        stage1_teacher_cls,
        image,
        stage1_teacher_prob,
        stage1_prediction,
        args,
        cfg,
        args.display_probe_strength,
        probe_seed,
    )

    normalized_key = normalized_history_key(sample["name"])
    checkpoint_key = temporal_keys.get(normalized_key)
    if checkpoint_key is None:
        raise KeyError("No Stage-2 temporal snapshot for {}".format(sample["name"]))
    hist_label, hist_conf = resize_history(
        temporal_history[checkpoint_key], stage2_prob.shape[-2:], device
    )
    temporal_raw = _compute_temporal_label_raw(
        cfg, stage2_prob, hist_label, hist_conf
    )
    temporal_reliability = float(args.temporal_floor) + (
        1.0 - float(args.temporal_floor)
    ) * temporal_raw

    stage2_training_state = stage2_state.get("training_state", {})
    class_prior = stage2_training_state.get("running_conf")
    if class_prior is None:
        raise KeyError("Stage-2 checkpoint lacks training_state.running_conf")
    class_prior = torch.as_tensor(class_prior, device=device).float()
    class_balance = _compute_class_balance_reliability(
        cfg, stage2_prediction, class_prior
    )
    temporal_class_weight = temporal_reliability * class_balance

    stage2_uwrg_method = uwrg_maps(
        stage2_teacher_fea,
        stage2_teacher_cls,
        image,
        stage2_prob,
        stage2_prediction,
        args,
        cfg,
        args.method_probe_strength,
        probe_seed,
    )
    final_weight_with_uwrg = temporal_class_weight * stage2_uwrg_method["reliability"]

    miou, per_class = sample_miou(
        stage2_prediction,
        label,
        cfg.MODEL.NUM_CLASSES,
        cfg.INPUT.IGNORE_LABEL,
    )
    err = error_fraction(stage2_prediction, label, cfg.INPUT.IGNORE_LABEL)
    original_image = _image_space(image, cfg)
    method_probe_image = stage1_uwrg_method["probe"]
    display_probe_image = stage1_uwrg_display["probe"]

    return {
        "image_normalized": image,
        "image": original_image,
        "ground_truth": label,
        "stage1_prediction": stage1_prediction,
        "stage2_prediction": stage2_prediction,
        "entropy": entropy_kl["entropy"],
        "kl_normalized": entropy_kl["kl_normalized"],
        "entropy_kl_reliability": entropy_kl["entropy_kl_reliability"],
        "uwrg_probe_method": method_probe_image,
        "uwrg_probe_display": display_probe_image,
        "uwrg_probe_prediction_method": stage1_uwrg_method["probe_prediction"],
        "uwrg_probe_prediction_display": stage1_uwrg_display["probe_prediction"],
        "uwrg_reliability_method": stage1_uwrg_method["reliability"],
        "uwrg_downweight_method": stage1_uwrg_method["downweight"],
        "uwrg_reliability_display": stage1_uwrg_display["reliability"],
        "uwrg_downweight_display": stage1_uwrg_display["downweight"],
        "temporal_history_label": hist_label,
        "temporal_agreement": (stage2_prediction.unsqueeze(1) == hist_label).float(),
        "temporal_reliability": temporal_reliability,
        "class_balance": class_balance,
        "temporal_class_weight": temporal_class_weight,
        "final_weight_with_uwrg": final_weight_with_uwrg,
        "stage2_uwrg_reliability": stage2_uwrg_method["reliability"],
        "miou": miou,
        "per_class_iou": per_class,
        "error_fraction": err,
        "probe_delta_method": float(
            torch.mean(torch.abs(method_probe_image - original_image)).item()
        ),
        "probe_delta_display": float(
            torch.mean(torch.abs(display_probe_image - original_image)).item()
        ),
        "uncertainty_stats_fallback": not all(
            uncertainty_stats.get(key) is not None
            for key in ("entropy_mean", "entropy_var", "kl_mean", "kl_var")
        ),
        "history_checkpoint_key": checkpoint_key,
    }


def clarity_record(sample: Dict[str, Any], maps: Dict[str, Any]) -> Dict[str, Any]:
    entropy_stats = map_stats(maps["entropy_kl_reliability"])
    uwrg_stats = map_stats(maps["uwrg_downweight_method"])
    temporal_stats = map_stats(maps["temporal_class_weight"])
    return {
        "index": sample["index"],
        "name": sample["name"],
        "stem": sample["stem"],
        "per_image_miou": maps["miou"],
        "error_fraction": maps["error_fraction"],
        "entropy_reliability_std": entropy_stats["std"],
        "uwrg_downweight_p99": uwrg_stats["p99"],
        "temporal_class_weight_std": temporal_stats["std"],
        "probe_delta_method": maps["probe_delta_method"],
        "probe_delta_display": maps["probe_delta_display"],
    }


def save_rgb(path: Path, image: np.ndarray) -> None:
    array = np.asarray(np.clip(image, 0.0, 1.0) * 255.0, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_label(path: Path, label: np.ndarray) -> None:
    Image.fromarray(label_rgb(label), mode="RGB").save(path)


def save_map(
    path: Path,
    values: np.ndarray,
    key: str,
    args: argparse.Namespace,
) -> None:
    vmin, vmax, cmap = MAP_SPECS[key]
    if key.startswith("uwrg_downweight"):
        vmax = max(1e-6, 1.0 - args.uwrg_floor ** args.uwrg_lambda)
    elif key == "class_balance":
        vmin, vmax = args.class_balance_min, args.class_balance_max
    elif key in ("temporal_class_weight", "final_weight_with_uwrg"):
        vmin, vmax = 0.0, args.class_balance_max

    absolute_path = path.with_name(path.stem + "_absolute" + path.suffix)
    plt.imsave(str(absolute_path), values, cmap=cmap, vmin=vmin, vmax=vmax)

    display_vmin, display_vmax = vmin, vmax
    if args.map_rendering == "contrast" and key != "temporal_agreement":
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = np.percentile(finite, (1.0, 99.0))
            if float(high - low) > 1e-6:
                display_vmin, display_vmax = float(low), float(high)
    plt.imsave(
        str(path),
        values,
        cmap=cmap,
        vmin=display_vmin,
        vmax=display_vmax,
    )


def save_error(path: Path, prediction: np.ndarray, label: np.ndarray, ignore: int) -> None:
    valid = label != ignore
    error = np.zeros((*label.shape, 3), dtype=np.uint8)
    error[valid & (prediction == label)] = (235, 235, 235)
    error[valid & (prediction != label)] = (230, 45, 45)
    error[~valid] = (90, 90, 90)
    Image.fromarray(error, mode="RGB").save(path)


def save_sample(
    output_dir: Path,
    rank: int,
    sample: Dict[str, Any],
    maps: Dict[str, Any],
    cfg: Any,
    args: argparse.Namespace,
    selection_row: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    sample_dir = output_dir / "{:02d}_{}".format(rank, sample["stem"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    image = tensor_image(maps["image"])
    gt = tensor_map(maps["ground_truth"]).astype(np.int64)
    stage1_prediction = tensor_map(maps["stage1_prediction"]).astype(np.int64)
    stage2_prediction = tensor_map(maps["stage2_prediction"]).astype(np.int64)
    history_label = tensor_map(maps["temporal_history_label"]).astype(np.int64)

    save_rgb(sample_dir / "image.png", image)
    save_label(sample_dir / "ground_truth.png", gt)
    save_label(sample_dir / "stage1_prediction.png", stage1_prediction)
    save_label(sample_dir / "stage2_prediction.png", stage2_prediction)
    save_label(sample_dir / "temporal_history_prediction.png", history_label)
    save_label(
        sample_dir / "uwrg_probe_prediction_method.png",
        tensor_map(maps["uwrg_probe_prediction_method"]).astype(np.int64),
    )
    save_label(
        sample_dir / "uwrg_probe_prediction_display.png",
        tensor_map(maps["uwrg_probe_prediction_display"]).astype(np.int64),
    )
    save_rgb(sample_dir / "uwrg_probe_method.png", tensor_image(maps["uwrg_probe_method"]))
    save_rgb(
        sample_dir / "uwrg_probe_display.png", tensor_image(maps["uwrg_probe_display"])
    )
    save_error(
        sample_dir / "stage2_error.png",
        stage2_prediction,
        gt,
        cfg.INPUT.IGNORE_LABEL,
    )

    for key in MAP_SPECS:
        values = tensor_map(maps[key])
        save_map(sample_dir / (key + ".png"), values, key, args)
        if args.save_npy:
            np.save(sample_dir / (key + ".npy"), values.astype(np.float32))

    metadata = {
        "name": sample["name"],
        "dataset_index": sample["index"],
        "selection_rank": rank,
        "selection": selection_row,
        "per_image_miou": maps["miou"],
        "per_present_class_iou": maps["per_class_iou"],
        "error_fraction": maps["error_fraction"],
        "selection_used_ground_truth": True,
        "history_checkpoint_key": maps["history_checkpoint_key"],
        "uncertainty_stats_fallback": maps["uncertainty_stats_fallback"],
        "method_probe_strength": args.method_probe_strength,
        "display_probe_strength": args.display_probe_strength,
        "display_probe_is_diagnostic_only": True,
        "uwrg_floor": args.uwrg_floor,
        "uwrg_lambda": args.uwrg_lambda,
        "main_png_map_rendering": args.map_rendering,
        "absolute_scale_suffix": "_absolute.png",
        "map_statistics": {
            key: map_stats(maps[key]) for key in MAP_SPECS
        },
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return {
        "Image": image,
        "GT": label_rgb(gt),
        "Stage-2 pred.": label_rgb(stage2_prediction),
        "Entropy": tensor_map(maps["entropy"]),
        "Entropy-KL rel.": tensor_map(maps["entropy_kl_reliability"]),
        "UWRG probe": tensor_image(maps["uwrg_probe_method"]),
        "UWRG rel.": tensor_map(maps["uwrg_reliability_method"]),
        "Temporal agr.": tensor_map(maps["temporal_agreement"]),
        "Class balance": tensor_map(maps["class_balance"]),
        "Stage-2 weight": tensor_map(maps["temporal_class_weight"]),
    }


def save_overview(
    path: Path,
    rows: Sequence[Tuple[str, Dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> None:
    if not rows:
        return
    columns = list(rows[0][1].keys())
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(2.8 * len(columns), 2.0 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (name, panels) in enumerate(rows):
        for column_index, title in enumerate(columns):
            ax = axes[row_index, column_index]
            values = panels[title]
            if values.ndim == 3:
                ax.imshow(values)
            elif title == "Temporal agr.":
                ax.imshow(values, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                finite = values[np.isfinite(values)]
                low, high = (0.0, 1.0)
                if finite.size:
                    p01, p99 = np.percentile(finite, (1.0, 99.0))
                    if float(p99 - p01) > 1e-6:
                        low, high = float(p01), float(p99)
                ax.imshow(values, cmap="gray", vmin=low, vmax=high)
            if row_index == 0:
                ax.set_title(title, fontsize=10)
            if column_index == 0:
                ax.set_ylabel(name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_readme(output_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# ReMask visualization assets: {args.setting_name}

This folder was generated post hoc. It does not modify checkpoints or reported
results. Samples were selected from `{args.dataset}` using ground-truth
per-image mIoU only for visualization. This selection must not be described as
an unbiased quantitative evaluation.

## Which files correspond to each stage

- `stage1_prediction.png`, `entropy.png`, `kl_normalized.png`, and
  `entropy_kl_reliability.png` come from the Stage-1 teacher/student checkpoint.
- `uwrg_probe_method.png` and `uwrg_reliability_method.png` use the exact UWRG
  strength ({args.method_probe_strength}) and floor ({args.uwrg_floor}) of this run.
- `uwrg_probe_display.png` uses strength {args.display_probe_strength} only to make
  attenuation/backscatter easier to see. It is diagnostic-only and must not be
  called the training perturbation without this qualification.
- `stage2_prediction.png`, `temporal_history_prediction.png`,
  `temporal_agreement.png`, `temporal_reliability.png`, `class_balance.png`, and
  `temporal_class_weight.png` come from the final Stage-2 checkpoint and its saved
  temporal history/class prior.
- `final_weight_with_uwrg.png` additionally multiplies the Stage-2 temporal-class
  weight by the Stage-2 UWRG reliability gate.

`overview.png` is a quick contact sheet. Individual PNGs have identical spatial
dimensions and are intended for assembling the final paper figure. Scalar maps
use black=low and white=high. Main PNGs use `{args.map_rendering}` rendering for
visibility; every `*_absolute.png` preserves the fixed physical/reliability
scale, and optional NPY files preserve the raw values exactly.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.dataset != "suim_train":
        raise ValueError(
            "Exact Stage-2 temporal maps require suim_train; received {}.".format(
                args.dataset
            )
        )
    if not (1 <= args.num_select <= args.candidate_pool):
        raise ValueError("Require 1 <= num-select <= candidate-pool.")
    if not (0.0 <= args.quality_weight <= 1.0):
        raise ValueError("quality-weight must be in [0,1].")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = build_cfg(args)

    stage1_path = resolve_path(args.stage1_checkpoint)
    stage2_path = resolve_path(args.stage2_checkpoint)
    output_dir = resolve_path(args.output_dir) / safe_stem(args.setting_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_readme(output_dir, args)

    print("Loading checkpoints...", flush=True)
    stage1_state = load_torch_checkpoint(stage1_path)
    stage2_state = load_torch_checkpoint(stage2_path)
    stage1_teacher = build_checkpoint_model(cfg, stage1_state, "teacher", device)
    stage1_student = build_checkpoint_model(cfg, stage1_state, "student", device)
    stage2_teacher = build_checkpoint_model(cfg, stage2_state, "teacher", device)
    stage1_models = (*stage1_teacher, *stage1_student)

    stage2_training_state = stage2_state.get("training_state", {})
    temporal_history = stage2_training_state.get("temporal_label_history", {})
    if not temporal_history:
        raise KeyError(
            "Stage-2 checkpoint has no temporal_label_history. Use the actual "
            "Stage-2 training checkpoint, not an inference-only export."
        )
    temporal_keys = history_index(temporal_history)
    print("Temporal snapshots: {}".format(len(temporal_history)), flush=True)

    dataset = make_dataset(cfg, args.dataset)
    print("Scanning {} samples for final Stage-2 per-image mIoU...".format(len(dataset)), flush=True)
    scan_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = get_sample(dataset, index, args)
            if normalized_history_key(sample["name"]) not in temporal_keys:
                continue
            image = sample["image"].to(device)
            label = sample["label"].to(device)
            probability = forward_prob(stage2_teacher[0], stage2_teacher[1], image)
            prediction = probability.argmax(dim=1)
            miou, _per_class = sample_miou(
                prediction,
                label,
                cfg.MODEL.NUM_CLASSES,
                cfg.INPUT.IGNORE_LABEL,
            )
            err = error_fraction(prediction, label, cfg.INPUT.IGNORE_LABEL)
            gt_classes = present_classes(label, cfg.INPUT.IGNORE_LABEL)
            scan_rows.append(
                {
                    "index": index,
                    "name": sample["name"],
                    "stem": sample["stem"],
                    "per_image_miou": miou,
                    "error_fraction": err,
                    "num_gt_classes": len(gt_classes),
                    "num_gt_foreground_classes": sum(
                        class_id != 0 for class_id in gt_classes
                    ),
                    "gt_classes": "|".join(str(class_id) for class_id in gt_classes),
                }
            )
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print("  scanned {}/{}".format(index + 1, len(dataset)), flush=True)

    scan_rows.sort(key=lambda row: row["per_image_miou"], reverse=True)
    for rank, row in enumerate(scan_rows, start=1):
        row["global_miou_rank"] = rank
    atomic_csv(output_dir / "all_sample_miou.csv", scan_rows)

    filtered = [
        row
        for row in scan_rows
        if args.min_error_fraction <= row["error_fraction"] <= args.max_error_fraction
        and row["num_gt_classes"] >= args.min_present_classes
        and row["num_gt_foreground_classes"] >= args.min_foreground_classes
    ]
    if len(filtered) < args.num_select:
        raise RuntimeError(
            "Only {} samples pass the error-fraction filter; need {}.".format(
                len(filtered), args.num_select
            )
        )
    candidates = filtered[: min(args.candidate_pool, len(filtered))]
    print(
        "Computing diagnostic clarity for top {} eligible samples...".format(
            len(candidates)
        ),
        flush=True,
    )
    clarity_rows: List[Dict[str, Any]] = []
    candidate_by_index = {int(row["index"]): row for row in candidates}
    for candidate_index, row in enumerate(candidates, start=1):
        sample = get_sample(dataset, int(row["index"]), args)
        maps = compute_maps(
            sample,
            cfg,
            args,
            device,
            stage1_models,
            stage2_teacher,
            stage1_state,
            stage2_state,
            temporal_history,
            temporal_keys,
        )
        record = clarity_record(sample, maps)
        record["global_miou_rank"] = row["global_miou_rank"]
        clarity_rows.append(record)
        print(
            "  diagnostics {}/{}: {} mIoU={:.4f}".format(
                candidate_index, len(candidates), sample["name"], maps["miou"]
            ),
            flush=True,
        )

    quality_rank = percentile_rank(
        [row["per_image_miou"] for row in clarity_rows], higher_is_better=True
    )
    clarity_components = np.stack(
        [
            percentile_rank(
                [row["entropy_reliability_std"] for row in clarity_rows], True
            ),
            percentile_rank([row["uwrg_downweight_p99"] for row in clarity_rows], True),
            percentile_rank(
                [row["temporal_class_weight_std"] for row in clarity_rows], True
            ),
            percentile_rank([row["probe_delta_method"] for row in clarity_rows], True),
        ],
        axis=1,
    )
    clarity_score = clarity_components.mean(axis=1)
    for index, row in enumerate(clarity_rows):
        row["quality_percentile"] = float(quality_rank[index])
        row["clarity_score"] = float(clarity_score[index])
        row["balanced_score"] = float(
            args.quality_weight * quality_rank[index]
            + (1.0 - args.quality_weight) * clarity_score[index]
        )

    if args.selection_mode == "top_miou":
        selected_rows = sorted(
            clarity_rows, key=lambda row: row["per_image_miou"], reverse=True
        )[: args.num_select]
    else:
        selected_rows = sorted(
            clarity_rows, key=lambda row: row["balanced_score"], reverse=True
        )[: args.num_select]
    selected_rows.sort(key=lambda row: row["per_image_miou"], reverse=True)
    for rank, row in enumerate(selected_rows, start=1):
        row["selection_rank"] = rank

    atomic_csv(output_dir / "candidate_diagnostics.csv", clarity_rows)
    atomic_csv(output_dir / "selected_samples.csv", selected_rows)

    print("Saving {} selected samples...".format(len(selected_rows)), flush=True)
    overview_rows: List[Tuple[str, Dict[str, np.ndarray]]] = []
    for rank, row in enumerate(selected_rows, start=1):
        sample = get_sample(dataset, int(row["index"]), args)
        maps = compute_maps(
            sample,
            cfg,
            args,
            device,
            stage1_models,
            stage2_teacher,
            stage1_state,
            stage2_state,
            temporal_history,
            temporal_keys,
        )
        panels = save_sample(output_dir, rank, sample, maps, cfg, args, row)
        overview_rows.append(
            ("{}\nIoU {:.1f}".format(sample["stem"], 100.0 * maps["miou"]), panels)
        )
    save_overview(output_dir / "overview.png", overview_rows, args)

    run_metadata = {
        "setting_name": args.setting_name,
        "dataset": args.dataset,
        "dataset_size": len(dataset),
        "num_temporal_snapshots": len(temporal_history),
        "stage1_checkpoint": str(stage1_path),
        "stage2_checkpoint": str(stage2_path),
        "selection_mode": args.selection_mode,
        "selection_used_ground_truth": True,
        "num_select": args.num_select,
        "candidate_pool": len(candidates),
        "method_probe_strength": args.method_probe_strength,
        "display_probe_strength": args.display_probe_strength,
        "display_probe_is_diagnostic_only": True,
        "uwrg_floor": args.uwrg_floor,
        "uwrg_lambda": args.uwrg_lambda,
        "temporal_floor": args.temporal_floor,
        "class_balance_gamma": args.class_balance_gamma,
        "class_balance_range": [args.class_balance_min, args.class_balance_max],
        "resolution": [args.width, args.height],
        "seed": args.seed,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print("Done: {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
