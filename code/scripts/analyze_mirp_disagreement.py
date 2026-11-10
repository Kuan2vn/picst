#!/usr/bin/env python
"""Test whether MIRP reconstruction disagreement predicts pseudo-label errors.

The source segmentation model produces ``p_T`` on a clean Cityscapes image.
For ``p_MIRP``, the same frozen source encoder sees multiple masked versions of
the image and the trained MIRP predictor reconstructs the clean latent.  Only
predictions at masked pixels are averaged.  Cityscapes labels are read after
both predictions have been produced and are used exclusively for analysis.
"""

import argparse
import csv
import json
import math
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_classifier, build_feature_extractor
from core.utils.misc import strip_prefix_if_present
from train_MIRP import _apply_input_mask, _build_mirp_predictor, _sample_feature_mask


EPS = 1e-8
METRIC_LIMITS = {"js": math.log(2.0), "cosine": 1.0}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIRP disagreement versus Cityscapes pseudo-label accuracy"
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="cityscapes_val")
    parser.add_argument("--predictor-type", default="spatial3x3")
    parser.add_argument("--predictor-hidden-dim", type=int, default=256)
    parser.add_argument("--mask-samples", type=int, default=32)
    parser.add_argument("--mask-batch-size", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0, help="0 uses the full split")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-pixels", type=int, default=2_000_000)
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument("--fixed-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_state(module, state):
    module.load_state_dict(strip_prefix_if_present(state, "module."), strict=False)


def js_divergence(p, q):
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())).sum(dim=1)
        + (q * (q.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())).sum(dim=1)
    )


def cosine_disagreement(p, q):
    similarity = (p * q).sum(dim=1) / (
        p.square().sum(dim=1).sqrt() * q.square().sum(dim=1).sqrt()
    ).clamp_min(EPS)
    return (1.0 - similarity).clamp(0.0, 1.0)


def quantile_rows(metric, values, correct, bin_count):
    order = np.argsort(values, kind="mergesort")
    rows = []
    for index, indices in enumerate(np.array_split(order, bin_count), start=1):
        if indices.size == 0:
            continue
        bin_values = values[indices]
        accuracy = float(correct[indices].mean())
        rows.append(
            {
                "metric": metric,
                "bin": index,
                "count": int(indices.size),
                "disagreement_low": float(bin_values.min()),
                "disagreement_high": float(bin_values.max()),
                "disagreement_mean": float(bin_values.mean()),
                "pseudo_label_accuracy": accuracy,
                "pseudo_label_error_rate": 1.0 - accuracy,
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path, rows):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    titles = {"js": "Jensen-Shannon disagreement", "cosine": "Cosine disagreement"}
    for axis, metric in zip(axes, ("js", "cosine")):
        selected = [row for row in rows if row["metric"] == metric]
        x = [row["bin"] for row in selected]
        y = [100.0 * row["pseudo_label_accuracy"] for row in selected]
        axis.plot(x, y, marker="o", linewidth=2.2, color="#1769aa")
        axis.set_title(titles[metric])
        axis.set_xlabel("Disagreement quantile (low → high)")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Source pseudo-label accuracy (%)")
    figure.suptitle("GTA5 source model on Cityscapes: MIRP disagreement vs. correctness")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.mask_samples <= 0 or args.mask_batch_size <= 0:
        raise ValueError("--mask-samples and --mask-batch-size must be positive")
    if args.quantile_bins < 2 or args.fixed_bins < 2:
        raise ValueError("At least two analysis bins are required")
    if args.sample_pixels <= 0:
        raise ValueError("--sample-pixels must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.TEST = args.dataset
    cfg.MIRP.PREDICTOR_TYPE = args.predictor_type
    cfg.MIRP.PREDICTOR_HIDDEN_DIM = args.predictor_hidden_dim
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    required = ("feature_extractor", "classifier", "mirp_predictor")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError("MIRP checkpoint is missing keys: {}".format(missing))

    feature_extractor = build_feature_extractor(cfg).to(device)
    classifier = build_classifier(cfg).to(device)
    predictor = _build_mirp_predictor(cfg, classifier).to(device)
    load_state(feature_extractor, checkpoint["feature_extractor"])
    load_state(classifier, checkpoint["classifier"])
    load_state(predictor, checkpoint["mirp_predictor"])
    feature_extractor.eval()
    classifier.eval()
    predictor.eval()

    dataset = build_dataset(cfg, mode="val", is_source=False)
    total_images = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))
    if total_images <= 0:
        raise RuntimeError("The selected Cityscapes split contains no images")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    samples_per_image = max(1, int(math.ceil(args.sample_pixels / float(total_images))))

    fixed = {
        metric: {
            "count": np.zeros(args.fixed_bins, dtype=np.int64),
            "correct": np.zeros(args.fixed_bins, dtype=np.float64),
            "disagreement": np.zeros(args.fixed_bins, dtype=np.float64),
        }
        for metric in METRIC_LIMITS
    }
    sampled = {"js": [], "cosine": [], "correct": [], "confidence": []}
    valid_pixel_count = 0
    gt_valid_pixel_count = 0
    correct_pixel_count = 0
    covered_pixel_count = 0

    def accumulate_masked_predictions(image, feature_size, probability_sum, coverage, count):
        feature_masks = _sample_feature_mask(
            cfg, count, feature_size[0], feature_size[1], device
        )
        image_batch = image.expand(count, -1, -1, -1)
        masked_images, image_masks = _apply_input_mask(cfg, image_batch, feature_masks)
        _, masked_features = feature_extractor(masked_images)
        reconstructed_features = predictor(masked_features)
        reconstructed_logits = classifier(reconstructed_features, image.shape[-2:])
        reconstructed_probability = F.softmax(reconstructed_logits, dim=1)
        probability_sum.add_((reconstructed_probability * image_masks).sum(dim=0, keepdim=True))
        coverage.add_(image_masks.sum(dim=0, keepdim=True))

    with torch.inference_mode():
        for image_index, (image, ground_truth, _name) in enumerate(loader):
            if image_index >= total_images:
                break
            image = image.to(device, non_blocking=True)
            ground_truth = ground_truth.to(device, non_blocking=True).long()
            input_size = image.shape[-2:]

            _, clean_features = feature_extractor(image)
            clean_logits = classifier(clean_features, input_size)
            clean_probability = F.softmax(clean_logits, dim=1)
            feature_size = clean_features.shape[-2:]
            del clean_logits, clean_features

            probability_sum = torch.zeros_like(clean_probability)
            coverage = clean_probability.new_zeros((1, 1, *input_size))
            completed = 0
            while completed < args.mask_samples:
                count = min(args.mask_batch_size, args.mask_samples - completed)
                accumulate_masked_predictions(
                    image, feature_size, probability_sum, coverage, count
                )
                completed += count

            # Random block masks can leave a few cells uncovered.  One or two
            # targeted passes make p_MIRP defined for every evaluated pixel.
            for _ in range(2):
                uncovered = coverage.eq(0)
                if not uncovered.any():
                    break
                feature_mask = F.interpolate(
                    uncovered.float(), size=feature_size, mode="nearest"
                )
                masked_image, image_mask = _apply_input_mask(cfg, image, feature_mask)
                _, masked_features = feature_extractor(masked_image)
                reconstructed = predictor(masked_features)
                reconstructed_probability = F.softmax(
                    classifier(reconstructed, input_size), dim=1
                )
                probability_sum.add_(reconstructed_probability * image_mask)
                coverage.add_(image_mask)

            reconstructed_probability = probability_sum / coverage.clamp_min(1.0)
            gt_resized = F.interpolate(
                ground_truth.unsqueeze(1).float(), size=input_size, mode="nearest"
            ).squeeze(1).long()
            pseudo_label = clean_probability.argmax(dim=1)
            confidence = clean_probability.max(dim=1).values
            correct = pseudo_label.eq(gt_resized)
            valid = gt_resized.ne(cfg.INPUT.IGNORE_LABEL) & coverage.squeeze(1).gt(0)

            disagreements = {
                "js": js_divergence(clean_probability, reconstructed_probability),
                "cosine": cosine_disagreement(
                    clean_probability, reconstructed_probability
                ),
            }
            valid_count = int(valid.sum().item())
            gt_valid_pixel_count += int(gt_resized.ne(cfg.INPUT.IGNORE_LABEL).sum().item())
            valid_pixel_count += valid_count
            correct_pixel_count += int((correct & valid).sum().item())
            covered_pixel_count += int(
                (gt_resized.ne(cfg.INPUT.IGNORE_LABEL) & coverage.squeeze(1).gt(0)).sum().item()
            )

            valid_correct = correct[valid].float()
            for metric, disagreement in disagreements.items():
                values = disagreement[valid].clamp(0.0, METRIC_LIMITS[metric])
                bin_index = torch.floor(
                    values / METRIC_LIMITS[metric] * args.fixed_bins
                ).long().clamp(0, args.fixed_bins - 1)
                fixed[metric]["count"] += torch.bincount(
                    bin_index, minlength=args.fixed_bins
                ).cpu().numpy()
                fixed[metric]["correct"] += torch.bincount(
                    bin_index,
                    weights=valid_correct,
                    minlength=args.fixed_bins,
                ).cpu().numpy()
                fixed[metric]["disagreement"] += torch.bincount(
                    bin_index,
                    weights=values,
                    minlength=args.fixed_bins,
                ).cpu().numpy()

            if valid_count > 0:
                valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
                take = min(samples_per_image, valid_count)
                chosen = valid_indices[
                    torch.randint(valid_count, (take,), device=device)
                ]
                sampled["js"].append(disagreements["js"].reshape(-1)[chosen].cpu().numpy())
                sampled["cosine"].append(
                    disagreements["cosine"].reshape(-1)[chosen].cpu().numpy()
                )
                sampled["correct"].append(correct.reshape(-1)[chosen].byte().cpu().numpy())
                sampled["confidence"].append(
                    confidence.reshape(-1)[chosen].cpu().numpy()
                )

            if (image_index + 1) % 10 == 0 or image_index + 1 == total_images:
                print(
                    "Analyzed {}/{} images; source pseudo-label accuracy={:.2f}%".format(
                        image_index + 1,
                        total_images,
                        100.0 * correct_pixel_count / max(1, valid_pixel_count),
                    ),
                    flush=True,
                )

    arrays = {key: np.concatenate(value) for key, value in sampled.items()}
    if arrays["correct"].size > args.sample_pixels:
        keep = np.random.RandomState(args.seed).choice(
            arrays["correct"].size, args.sample_pixels, replace=False
        )
        arrays = {key: value[keep] for key, value in arrays.items()}
    sampled_correct = arrays["correct"].astype(np.float64)

    quantile_output = []
    metric_summaries = {}
    for metric in ("js", "cosine"):
        metric_rows = quantile_rows(
            metric, arrays[metric], sampled_correct, args.quantile_bins
        )
        quantile_output.extend(metric_rows)
        low_accuracy = metric_rows[0]["pseudo_label_accuracy"]
        high_accuracy = metric_rows[-1]["pseudo_label_accuracy"]
        drops = sum(
            right["pseudo_label_accuracy"] <= left["pseudo_label_accuracy"]
            for left, right in zip(metric_rows, metric_rows[1:])
        )
        metric_summaries[metric] = {
            "bottom_quantile_accuracy": low_accuracy,
            "top_quantile_accuracy": high_accuracy,
            "accuracy_drop_bottom_to_top": low_accuracy - high_accuracy,
            "non_increasing_adjacent_steps": int(drops),
            "adjacent_steps": max(0, len(metric_rows) - 1),
        }

    quantile_fields = [
        "metric",
        "bin",
        "count",
        "disagreement_low",
        "disagreement_high",
        "disagreement_mean",
        "pseudo_label_accuracy",
        "pseudo_label_error_rate",
    ]
    write_csv(
        os.path.join(args.output_dir, "disagreement_quantile_bins.csv"),
        quantile_output,
        quantile_fields,
    )

    fixed_rows = []
    for metric, limit in METRIC_LIMITS.items():
        width = limit / args.fixed_bins
        for index in range(args.fixed_bins):
            count = int(fixed[metric]["count"][index])
            fixed_rows.append(
                {
                    "metric": metric,
                    "bin": index + 1,
                    "disagreement_low": index * width,
                    "disagreement_high": (index + 1) * width,
                    "count": count,
                    "disagreement_mean": (
                        fixed[metric]["disagreement"][index] / count if count else ""
                    ),
                    "pseudo_label_accuracy": (
                        fixed[metric]["correct"][index] / count if count else ""
                    ),
                }
            )
    write_csv(
        os.path.join(args.output_dir, "disagreement_fixed_bins.csv"),
        fixed_rows,
        [
            "metric",
            "bin",
            "disagreement_low",
            "disagreement_high",
            "count",
            "disagreement_mean",
            "pseudo_label_accuracy",
        ],
    )
    plot_curves(
        os.path.join(
            args.output_dir,
            "reconstruction_disagreement_vs_pseudolabel_accuracy.png",
        ),
        quantile_output,
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "sampled_pixel_probe.npz"), **arrays
    )

    summary = {
        "config_file": os.path.abspath(args.config_file),
        "checkpoint": os.path.abspath(args.checkpoint),
        "dataset": args.dataset,
        "images": total_images,
        "valid_pixels": valid_pixel_count,
        "gt_valid_pixels": gt_valid_pixel_count,
        "sampled_pixels": int(arrays["correct"].size),
        "source_pseudo_label_accuracy": correct_pixel_count / max(1, valid_pixel_count),
        "mask_samples": args.mask_samples,
        "covered_valid_pixels": covered_pixel_count,
        "valid_pixel_coverage": covered_pixel_count / max(1, gt_valid_pixel_count),
        "predictor_type": args.predictor_type,
        "predictor_hidden_dim": args.predictor_hidden_dim,
        "metrics": metric_summaries,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
