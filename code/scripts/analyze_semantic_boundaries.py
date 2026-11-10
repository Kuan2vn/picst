#!/usr/bin/env python3
"""Evaluate semantic boundary F1 from existing mask-inference exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PALETTE = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 0): 4,
    (255, 0, 255): 5,
    (255, 255, 0): 6,
    (255, 255, 255): 7,
    (128, 128, 128): 255,
}
CLASS_NAMES = ["Background", "Diver", "Plant", "Wreck", "Robot", "Reef", "Fish", "Floor"]
METHOD_LABELS = {
    "tt_sfuda": "TT-SFUDA",
    "cal_sfda": "Cal-SFDA",
    "sfmim": "SFMIM",
    "crossmatch": "CrossMatch",
    "uotsl": "UOTSL",
    "iapc": "IAPC",
    "snd_metapl": "SND",
    "remask": "ReMask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-root",
        default="validate_results/efficiency_uiis_to_suim/remask_high_gap_26496_mask_export",
    )
    parser.add_argument("--tolerance", type=int, default=3, help="Boundary matching radius in pixels.")
    parser.add_argument(
        "--output",
        default="validate_results/mask_boundary_quality/uiis_suim_boundary_f1.csv",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rgb_to_label(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    label = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in PALETTE.items():
        label[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = class_id
    return label


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (mask > eroded).astype(np.uint8)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_method(rows: Iterable[Dict[str, str]], tolerance: int) -> Tuple[np.ndarray, np.ndarray]:
    matched_pred = np.zeros(8, dtype=np.float64)
    pred_total = np.zeros(8, dtype=np.float64)
    matched_gt = np.zeros(8, dtype=np.float64)
    gt_total = np.zeros(8, dtype=np.float64)
    intersection = np.zeros(8, dtype=np.float64)
    union = np.zeros(8, dtype=np.float64)
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)

    for row in rows:
        pred = rgb_to_label(resolve(row["prediction"]))
        gt = rgb_to_label(resolve(row["ground_truth"]))
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = gt != 255
        for class_id in range(8):
            pred_mask = (pred == class_id) & valid
            gt_mask = (gt == class_id) & valid
            intersection[class_id] += np.logical_and(pred_mask, gt_mask).sum()
            union[class_id] += np.logical_or(pred_mask, gt_mask).sum()

            pred_boundary = inner_boundary(pred_mask)
            gt_boundary = inner_boundary(gt_mask)
            pred_count = float(pred_boundary.sum())
            gt_count = float(gt_boundary.sum())
            pred_total[class_id] += pred_count
            gt_total[class_id] += gt_count
            if pred_count > 0 and gt_count > 0:
                gt_band = cv2.dilate(gt_boundary, kernel, iterations=1) > 0
                pred_band = cv2.dilate(pred_boundary, kernel, iterations=1) > 0
                matched_pred[class_id] += np.logical_and(pred_boundary > 0, gt_band).sum()
                matched_gt[class_id] += np.logical_and(gt_boundary > 0, pred_band).sum()

    precision = np.divide(matched_pred, pred_total, out=np.zeros_like(matched_pred), where=pred_total > 0)
    recall = np.divide(matched_gt, gt_total, out=np.zeros_like(matched_gt), where=gt_total > 0)
    bf1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return bf1, iou


def main() -> None:
    args = parse_args()
    profile_root = resolve(args.profile_root)
    output = resolve(args.output)
    records = []
    for method_dir in sorted(path for path in profile_root.iterdir() if path.is_dir()):
        csv_path = method_dir / "mask_inference" / "mask_inference_samples.csv"
        if not csv_path.is_file():
            continue
        bf1, iou = evaluate_method(load_rows(csv_path), max(0, args.tolerance))
        record = {
            "method": method_dir.name,
            "method_label": METHOD_LABELS.get(method_dir.name, method_dir.name),
            "miou": float(iou.mean()),
            "mean_boundary_f1": float(bf1.mean()),
            "foreground_boundary_f1": float(bf1[1:].mean()),
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            record["{}_boundary_f1".format(class_name.lower())] = float(bf1[class_id])
        records.append(record)

    records.sort(key=lambda row: row["foreground_boundary_f1"], reverse=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print("Boundary tolerance: {} px".format(max(0, args.tolerance)))
    print("{:<12} {:>8} {:>10} {:>10}".format("Method", "mIoU", "mBF1", "FG-mBF1"))
    for row in records:
        print(
            "{:<12} {:>8.4f} {:>10.4f} {:>10.4f}".format(
                row["method_label"], row["miou"], row["mean_boundary_f1"], row["foreground_boundary_f1"]
            )
        )
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
