#!/usr/bin/env python3
"""Export a post-hoc visualization of the exact ReMask UWRG mechanism.

This script is diagnostic only: it reads one checkpoint and one target image,
runs the original and underwater-probed teacher predictions, and writes new
visualization files. It does not train, update, or overwrite model results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.configs.defaults import _C as DEFAULT_CFG
from core.datasets import build_dataset
from core.models import build_classifier
from core.models.feature_extractor import resnet_feature_extractor
from core.utils.misc import strip_prefix_if_present
from train_ReMask_special import (
    _compute_underwater_stability,
    _image_space,
    _normalized_space,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the exact underwater reliability gate used by ReMask."
    )
    parser.add_argument(
        "--config-file",
        default="configs/deeplabv2_r101_dtst_SUIM.yaml",
        help="Model/data config used by the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="pretrained/model_iter018000.pth",
        help="ReMask/source checkpoint containing feature_extractor and classifier.",
    )
    parser.add_argument(
        "--test-set",
        default="suim_val",
        help="DatasetCatalog target split to visualize.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.20)
    parser.add_argument("--floor", type=float, default=0.80)
    parser.add_argument(
        "--uwrg-lambda",
        type=float,
        default=1.0,
        help="Exponent in the weighted-product composition: effective_gate=gate^lambda.",
    )
    parser.add_argument("--mode", choices=("pixel", "segment"), default="segment")
    parser.add_argument("--segment-stride", type=int, default=4)
    parser.add_argument("--min-segment-pixels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        default="validate_results/uwrg_visualization",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional yacs overrides, e.g. INPUT.INPUT_SIZE_TEST '(960,480)'.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_cfg(args: argparse.Namespace) -> Any:
    cfg = DEFAULT_CFG.clone()
    cfg.merge_from_file(str(resolve_path(args.config_file)))
    cfg.merge_from_list(["DATASETS.TEST", args.test_set])
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def load_checkpoint(
    feature_extractor: torch.nn.Module,
    classifier: torch.nn.Module,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=torch.device("cpu"), weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))

    required = {"feature_extractor", "classifier"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(
            "Checkpoint {} is missing keys: {}".format(
                checkpoint_path, ", ".join(sorted(missing))
            )
        )
    feature_weights = strip_prefix_if_present(checkpoint["feature_extractor"], "module.")
    classifier_weights = strip_prefix_if_present(checkpoint["classifier"], "module.")
    feature_extractor.load_state_dict(feature_weights, strict=True)
    classifier.load_state_dict(classifier_weights, strict=True)
    return checkpoint


def build_checkpoint_feature_extractor(cfg: Any) -> torch.nn.Module:
    """Build the configured architecture without an unnecessary ImageNet load.

    Every parameter is immediately replaced by the supplied checkpoint, so the
    normal pretrained-backbone download in build_feature_extractor is both
    redundant and undesirable for a read-only offline diagnostic.
    """
    _model_name, backbone_name = cfg.MODEL.NAME.split("_")
    if not backbone_name.startswith("resnet"):
        raise NotImplementedError(
            "This diagnostic currently supports the ReMask ResNet checkpoints."
        )
    return resnet_feature_extractor(
        backbone_name,
        pretrained_weights=None,
        aux=True,
        pretrained_backbone=False,
        freeze_bn=cfg.MODEL.FREEZE_BN,
    )


def forward_prob(
    feature_extractor: torch.nn.Module,
    classifier: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    _, features = feature_extractor(images)
    logits = classifier(features, images.shape[-2:])
    return F.softmax(logits, dim=1)


def tensor_image(x: torch.Tensor) -> np.ndarray:
    return (
        x[0]
        .detach()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def tensor_map(x: torch.Tensor) -> np.ndarray:
    while x.ndim > 2:
        x = x[0]
    return x.detach().float().cpu().numpy()


def label_rgb(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.int64)
    rgb = np.full((*label.shape, 3), 128, dtype=np.uint8)
    valid = (label >= 0) & (label < len(SUIM_PALETTE))
    rgb[valid] = SUIM_PALETTE[label[valid]]
    return rgb


def safe_stem(name: Any) -> str:
    if isinstance(name, (list, tuple)):
        name = name[0] if name else "sample"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(name)).stem) or "sample"


def save_scalar(path: Path, values: np.ndarray, vmin: float, vmax: float, cmap: str) -> None:
    plt.imsave(str(path), values, vmin=vmin, vmax=vmax, cmap=cmap)


def add_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    artist = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.colorbar(artist, ax=ax, fraction=0.035, pad=0.02)


def save_panel(
    output_path: Path,
    original: np.ndarray,
    perturbed: np.ndarray,
    pred: np.ndarray,
    perturbed_pred: np.ndarray,
    agreement: np.ndarray,
    pixel_stability: np.ndarray,
    segment_stability: np.ndarray,
    downweight: np.ndarray,
    sample_name: str,
    strength: float,
    floor: float,
    uwrg_lambda: float,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels: Tuple[Tuple[np.ndarray, str], ...] = (
        (original, "(a) Target image"),
        (perturbed, "(b) Underwater probe"),
        (label_rgb(pred), "(c) Teacher prediction"),
        (label_rgb(perturbed_pred), "(d) Probe prediction"),
    )
    for ax, (image, title) in zip(axes[0], panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    add_heatmap(
        fig,
        axes[1, 0],
        agreement,
        "(e) Prediction agreement",
        "gray",
        0.0,
        1.0,
    )
    add_heatmap(
        fig,
        axes[1, 1],
        pixel_stability,
        "(f) Pixel stability",
        "viridis",
        0.0,
        1.0,
    )
    add_heatmap(
        fig,
        axes[1, 2],
        segment_stability,
        "(g) Segment-median stability",
        "viridis",
        0.0,
        1.0,
    )
    max_downweight = max(1e-6, 1.0 - float(floor) ** float(uwrg_lambda))
    add_heatmap(
        fig,
        axes[1, 3],
        downweight,
        r"(h) UWRG attenuation $1-g^{\lambda}$",
        "magma",
        0.0,
        max_downweight,
    )

    legend = [
        Patch(facecolor=SUIM_PALETTE[idx] / 255.0, edgecolor="0.45", label=name)
        for idx, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(
        handles=legend,
        loc="outside lower center",
        ncol=8,
        fontsize=9,
        frameon=False,
    )
    fig.suptitle(
        "UWRG diagnostic — {}  |  strength={:.2f}, floor={:.2f}, lambda={:.2f}".format(
            sample_name, strength, floor, uwrg_lambda
        ),
        fontsize=14,
    )
    fig.savefig(str(output_path), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_mask_analysis_panel(
    output_path: Path,
    original: np.ndarray,
    pred: np.ndarray,
    ground_truth: np.ndarray,
    entropy: np.ndarray,
    downweight: np.ndarray,
    error: np.ndarray,
    max_downweight: float,
) -> None:
    """Create an SND-style uncertainty-versus-oracle-error panel."""
    fig, axes = plt.subplots(1, 6, figsize=(15.5, 3.15))
    panels = (
        (original, "Image", None, None),
        (label_rgb(pred), "Prediction", None, None),
        (label_rgb(ground_truth), "Ground Truth", None, None),
        (entropy, "Entropy", "gray", (0.0, 1.0)),
        (downweight, "UWRG", "gray", (0.0, max_downweight)),
        (error, "Error vs. GT", "gray", (0.0, 1.0)),
    )
    for ax, (values, title, cmap, limits) in zip(axes, panels):
        if cmap is None:
            ax.imshow(values, interpolation="nearest")
        else:
            ax.imshow(
                values,
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
            )
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.axis("off")

    legend = [
        Patch(facecolor=SUIM_PALETTE[idx] / 255.0, edgecolor="0.45", label=name)
        for idx, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=8,
        fontsize=8.5,
        frameon=False,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.22, wspace=0.025)
    fig.savefig(str(output_path), dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.floor <= 1.0:
        raise ValueError("--floor must be in [0, 1].")
    if args.strength < 0.0 or args.uwrg_lambda < 0.0:
        raise ValueError("--strength and --uwrg-lambda must be non-negative.")

    cfg = build_cfg(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = build_checkpoint_feature_extractor(cfg).to(device)
    classifier = build_classifier(cfg).to(device)
    checkpoint = load_checkpoint(feature_extractor, classifier, checkpoint_path)
    feature_extractor.eval()
    classifier.eval()

    dataset = build_dataset(cfg, mode="test", is_source=False)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            "--sample-index {} is outside dataset range [0, {}).".format(
                args.sample_index, len(dataset)
            )
        )
    image, label, name = dataset[args.sample_index]
    image = image.unsqueeze(0).to(device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():
        original_prob = forward_prob(feature_extractor, classifier, image)
        original_image = _image_space(image, cfg)
        perturbed_image = _underwater_reliability_perturb(
            original_image, args.strength
        )
        perturbed_input = _normalized_space(perturbed_image, cfg)
        perturbed_prob = forward_prob(feature_extractor, classifier, perturbed_input)

        _confidence, pseudo_label = original_prob.max(dim=1)
        pixel_stability, agreement, _ = _compute_underwater_stability(
            original_prob,
            perturbed_prob,
            pseudo_label,
            cfg.INPUT.IGNORE_LABEL,
            "pixel",
            args.segment_stride,
            args.min_segment_pixels,
        )
        segment_stability, _, segment_count = _compute_underwater_stability(
            original_prob,
            perturbed_prob,
            pseudo_label,
            cfg.INPUT.IGNORE_LABEL,
            args.mode,
            args.segment_stride,
            args.min_segment_pixels,
        )
        raw_gate = (
            args.floor + (1.0 - args.floor) * segment_stability
        ).clamp(args.floor, 1.0)
        effective_gate = raw_gate.pow(args.uwrg_lambda)
        downweight = 1.0 - effective_gate
        perturbed_label = perturbed_prob.argmax(dim=1)
        normalized_entropy = (
            -(original_prob * original_prob.clamp_min(1e-8).log()).sum(dim=1)
            / np.log(max(int(cfg.MODEL.NUM_CLASSES), 2))
        ).clamp(0.0, 1.0)
        ground_truth = F.interpolate(
            label.unsqueeze(0).unsqueeze(0).float(),
            size=pseudo_label.shape[-2:],
            mode="nearest",
        ).squeeze(1).long().to(device)
        valid_ground_truth = ground_truth != int(cfg.INPUT.IGNORE_LABEL)
        prediction_error = (
            (pseudo_label != ground_truth) & valid_ground_truth
        ).float()

    stem = safe_stem(name)
    sample_dir = output_dir / "{:03d}_{}".format(args.sample_index, stem)
    sample_dir.mkdir(parents=True, exist_ok=True)

    original_np = tensor_image(original_image)
    perturbed_np = tensor_image(perturbed_image)
    pred_np = tensor_map(pseudo_label).astype(np.uint8)
    perturbed_pred_np = tensor_map(perturbed_label).astype(np.uint8)
    agreement_np = tensor_map(agreement)
    pixel_stability_np = tensor_map(pixel_stability)
    segment_stability_np = tensor_map(segment_stability)
    raw_gate_np = tensor_map(raw_gate)
    effective_gate_np = tensor_map(effective_gate)
    downweight_np = tensor_map(downweight)
    entropy_np = tensor_map(normalized_entropy)
    ground_truth_np = tensor_map(ground_truth).astype(np.uint8)
    prediction_error_np = tensor_map(prediction_error)

    Image.fromarray((original_np * 255.0).round().astype(np.uint8)).save(
        sample_dir / "target_image.png"
    )
    Image.fromarray((perturbed_np * 255.0).round().astype(np.uint8)).save(
        sample_dir / "underwater_probe.png"
    )
    Image.fromarray(label_rgb(pred_np)).save(sample_dir / "teacher_prediction.png")
    Image.fromarray(label_rgb(perturbed_pred_np)).save(
        sample_dir / "probe_prediction.png"
    )
    Image.fromarray(label_rgb(ground_truth_np)).save(sample_dir / "ground_truth.png")
    save_scalar(sample_dir / "agreement.png", agreement_np, 0.0, 1.0, "gray")
    save_scalar(
        sample_dir / "pixel_stability.png", pixel_stability_np, 0.0, 1.0, "viridis"
    )
    save_scalar(
        sample_dir / "segment_stability.png",
        segment_stability_np,
        0.0,
        1.0,
        "viridis",
    )
    save_scalar(sample_dir / "uwrg_gate.png", effective_gate_np, 0.0, 1.0, "viridis")
    save_scalar(
        sample_dir / "uwrg_attenuation.png",
        downweight_np,
        0.0,
        max(1e-6, 1.0 - args.floor ** args.uwrg_lambda),
        "magma",
    )
    save_scalar(sample_dir / "entropy.png", entropy_np, 0.0, 1.0, "gray")
    save_scalar(sample_dir / "error_vs_gt.png", prediction_error_np, 0.0, 1.0, "gray")

    panel_path = sample_dir / "uwrg_panel.png"
    save_panel(
        panel_path,
        original_np,
        perturbed_np,
        pred_np,
        perturbed_pred_np,
        agreement_np,
        pixel_stability_np,
        segment_stability_np,
        downweight_np,
        stem,
        args.strength,
        args.floor,
        args.uwrg_lambda,
    )
    mask_analysis_path = sample_dir / "mask_analysis_panel.png"
    save_mask_analysis_panel(
        mask_analysis_path,
        original_np,
        pred_np,
        ground_truth_np,
        entropy_np,
        downweight_np,
        prediction_error_np,
        max(1e-6, 1.0 - args.floor ** args.uwrg_lambda),
    )

    metadata = {
        "diagnostic_only": True,
        "sample_index": args.sample_index,
        "sample_name": str(name),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_iteration": checkpoint.get("iteration"),
        "config_file": str(resolve_path(args.config_file).resolve()),
        "test_set": args.test_set,
        "seed": args.seed,
        "strength": args.strength,
        "floor": args.floor,
        "uwrg_lambda": args.uwrg_lambda,
        "mode": args.mode,
        "segment_stride": args.segment_stride,
        "min_segment_pixels": args.min_segment_pixels,
        "segment_count": int(segment_count),
        "agreement_fraction": float(agreement_np.mean()),
        "disagreement_fraction": float(1.0 - agreement_np.mean()),
        "pixel_stability_mean": float(pixel_stability_np.mean()),
        "segment_stability_mean": float(segment_stability_np.mean()),
        "raw_gate_mean": float(raw_gate_np.mean()),
        "effective_gate_mean": float(effective_gate_np.mean()),
        "attenuation_mean": float(downweight_np.mean()),
        "error_fraction_excluding_ignore": float(
            prediction_error[valid_ground_truth].mean().item()
        ),
        "panel": str(panel_path.resolve()),
        "mask_analysis_panel": str(mask_analysis_path.resolve()),
    }
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
