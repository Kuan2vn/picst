#!/usr/bin/env python
"""Visualize online teacher-mask alignment into the student/strong view.

This is a debugging script for train_usis10k_instance_remask.py --online.
It runs the teacher on the original image, applies the exact strong transform
used by OnlinePseudoDatasetMapper to teacher masks, and writes side-by-side
overlays for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_usis10k_instance_remask as usis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usis-root", default="data/USIS10K/USIS10K")
    parser.add_argument("--label-mode", choices=("multi", "foreground"), default="multi")
    parser.add_argument("--backbone", choices=sorted(usis.BASIC_BACKBONES), default="mask_rcnn")
    parser.add_argument("--weights", required=True, help="Teacher checkpoint path.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output-dir", default="outputs/debug/mask_alignment")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--image-id", type=int, action="append", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def script_args_to_train_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        usis_root=args.usis_root,
        label_mode=args.label_mode,
        backbone=args.backbone,
        num_classes=None,
        teacher_weights=args.weights,
        weights=args.weights,
        device=args.device,
        opts=args.opts,
        num_workers=0,
        ims_per_batch=1,
        base_lr=2.5e-4,
        max_iter=1,
        eval_period=0,
        checkpoint_period=1000,
        pseudo_background_weight=1.0,
        mask_bce_weight=1.0,
        mask_dice_weight=1.0,
        no_ema=True,
        ema_decay=0.999,
        online=True,
        method="fixmatch",
        online_score_thresh=args.score_thresh,
        online_min_mask_area=args.min_mask_area,
        online_max_instances=args.max_instances,
        online_match_iou=0.5,
        online_rho=0.5,
        online_class_balance_prior="count",
        online_class_balance_momentum=0.9,
        online_gamma=0.25,
        online_w_min=0.7,
        online_w_max=1.3,
        allow_model_zoo_download=False,
    )


def chw_to_rgb_image(tensor: torch.Tensor, image_format: str) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {array.shape}")
    array = np.transpose(array, (1, 2, 0)).astype(np.uint8)
    if image_format.upper() == "BGR":
        array = array[:, :, ::-1]
    return np.ascontiguousarray(array)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay_masks(
    image_rgb: np.ndarray,
    masks: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
) -> Image.Image:
    base = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (230, 57, 70, 105),
        (29, 185, 84, 105),
        (30, 144, 255, 105),
        (255, 183, 3, 105),
        (156, 39, 176, 105),
        (0, 188, 212, 105),
    ]
    outlines = [
        (230, 57, 70, 230),
        (29, 185, 84, 230),
        (30, 144, 255, 230),
        (255, 183, 3, 230),
        (156, 39, 176, 230),
        (0, 188, 212, 230),
    ]
    for idx, mask in enumerate(masks):
        if mask.size == 0 or not mask.any():
            continue
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        color = Image.new("RGBA", base.size, colors[idx % len(colors)])
        overlay.alpha_composite(Image.composite(color, Image.new("RGBA", base.size), mask_img))
        bbox = mask_bbox(mask)
        if bbox is not None:
            draw.rectangle(bbox, outline=outlines[idx % len(outlines)], width=3)
            draw.text((bbox[0] + 3, max(0, bbox[1] - 14)), labels[idx], fill=outlines[idx % len(outlines)])

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    canvas = Image.new("RGB", (composed.width, composed.height + 28), (255, 255, 255))
    canvas.paste(composed, (0, 28))
    ImageDraw.Draw(canvas).text((8, 6), title, fill=(0, 0, 0))
    return canvas


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    height = max(left.height, right.height)
    width = left.width + right.width
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def selected_records(args: argparse.Namespace, train_args: argparse.Namespace) -> List[Dict[str, Any]]:
    ann_file = usis.annotation_path(train_args, args.split)
    image_root = usis.image_root(train_args, args.split)
    records = usis.load_unlabeled_coco_images(ann_file, image_root)
    if args.image_id:
        wanted = set(int(item) for item in args.image_id)
        records = [record for record in records if int(record["image_id"]) in wanted]
    else:
        rng = np.random.default_rng(args.seed)
        if len(records) > args.num_images:
            indices = rng.choice(len(records), size=args.num_images, replace=False)
            records = [records[int(index)] for index in indices]
        else:
            records = records[: args.num_images]
    return records


@torch.no_grad()
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_args = script_args_to_train_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d2 = usis.import_detectron2()
    cfg, model = usis.build_inference_model(d2, train_args)
    mapper = usis.OnlinePseudoDatasetMapper(cfg, is_train=True)
    records = selected_records(args, train_args)
    categories = usis.sorted_categories(usis.load_coco(usis.annotation_path(train_args, "train")))
    class_names = [str(cat["name"]) for cat in categories]

    summary_rows: List[Dict[str, Any]] = []
    for sample_idx, record in enumerate(records):
        mapped = mapper(record)
        teacher_input = {
            "image": mapped["teacher_image"],
            "height": int(mapped["height"]),
            "width": int(mapped["width"]),
        }
        outputs = model([teacher_input])[0]
        instances = outputs["instances"].to("cpu")
        teacher_rgb = chw_to_rgb_image(mapped["teacher_image"], cfg.INPUT.FORMAT)
        student_rgb = chw_to_rgb_image(mapped["image"], cfg.INPUT.FORMAT)
        transforms = mapped["strong_transforms"]

        original_masks: List[np.ndarray] = []
        strong_masks: List[np.ndarray] = []
        labels: List[str] = []
        if len(instances) > 0:
            scores = instances.scores.numpy()
            order = np.argsort(-scores).tolist()
            for rank in order:
                score = float(scores[rank])
                if score < args.score_thresh:
                    continue
                label_id = int(instances.pred_classes[rank])
                original_mask = instances.pred_masks[rank].numpy().astype(bool)
                strong_mask = usis.transform_binary_mask(original_mask, transforms)
                strong_area = int(strong_mask.sum())
                if strong_area < args.min_mask_area:
                    continue
                original_area = int(original_mask.sum())
                original_masks.append(original_mask)
                strong_masks.append(strong_mask)
                class_name = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
                labels.append(f"{len(labels)}:{class_name}:{score:.2f}")
                summary_rows.append(
                    {
                        "image_id": int(record["image_id"]),
                        "file_name": str(record["file_name"]),
                        "mask_index": len(labels) - 1,
                        "class_id": label_id,
                        "class_name": class_name,
                        "score": score,
                        "original_area": original_area,
                        "strong_area": strong_area,
                        "area_ratio": strong_area / max(original_area, 1),
                        "original_bbox_xyxy": json.dumps(mask_bbox(original_mask)),
                        "strong_bbox_xyxy": json.dumps(mask_bbox(strong_mask)),
                        "teacher_shape_hw": json.dumps(list(original_mask.shape)),
                        "student_shape_hw": json.dumps(list(strong_mask.shape)),
                        "transforms": str(transforms),
                    }
                )
                if len(labels) >= args.max_instances:
                    break

        teacher_panel = overlay_masks(
            teacher_rgb,
            original_masks,
            labels,
            f"teacher/original image_id={record['image_id']}",
        )
        student_panel = overlay_masks(
            student_rgb,
            strong_masks,
            labels,
            "student/strong view with transformed teacher masks",
        )
        combined = side_by_side(teacher_panel, student_panel)
        out_path = output_dir / f"alignment_{sample_idx:03d}_image_{record['image_id']}.png"
        combined.save(out_path)
        print(f"wrote {out_path} masks={len(labels)}")

    csv_path = output_dir / "alignment_summary.csv"
    fieldnames = [
        "image_id",
        "file_name",
        "mask_index",
        "class_id",
        "class_name",
        "score",
        "original_area",
        "strong_area",
        "area_ratio",
        "original_bbox_xyxy",
        "strong_bbox_xyxy",
        "teacher_shape_hw",
        "student_shape_hw",
        "transforms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"summary_csv={csv_path}")
    print(
        "Inspect the right panel in each PNG. The colored mask should cover the "
        "same object after resize/flip/crop as in the left teacher panel."
    )


if __name__ == "__main__":
    main()
