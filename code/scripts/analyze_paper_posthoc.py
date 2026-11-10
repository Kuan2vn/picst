#!/usr/bin/env python3
"""Compute three result-locked post-hoc analyses from exported SUIM masks.

Outputs:
  1. semantic Boundary IoU with the official 2% image-diagonal boundary width;
  2. paired image-level bootstrap confidence intervals for mIoU differences;
  3. pixel error transitions from the actual Stage-2 initialization to ReMask.

The script first reconstructs ordinary mIoU for every checkpoint and aborts on
any mismatch with the locked manifest.  It never replaces manuscript scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 8
IGNORE_LABEL = 255
CLASS_NAMES = ["Background", "Diver", "Plant", "Wreck", "Robot", "Reef", "Fish", "Floor"]
PALETTE = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 0): 4,
    (255, 0, 255): 5,
    (255, 255, 0): 6,
    (255, 255, 255): 7,
    (128, 128, 128): IGNORE_LABEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="scripts/paper_posthoc_checkpoints.csv")
    parser.add_argument("--export-root", default="validate_results/paper_posthoc_exports")
    parser.add_argument("--output-dir", default="validate_results/paper_posthoc_analysis")
    parser.add_argument("--boundary-ratio", type=float, default=0.02)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=8888)
    parser.add_argument("--miou-tolerance", type=float, default=7.5e-4)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] = ()) -> None:
    if not rows and not fields:
        return
    fieldnames = list(fields) if fields else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rgb_to_label(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    label = np.full(rgb.shape[:2], IGNORE_LABEL, dtype=np.uint8)
    for color, class_id in PALETTE.items():
        label[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = class_id
    unknown = (label == IGNORE_LABEL) & np.any(rgb != np.asarray((128, 128, 128), dtype=np.uint8), axis=2)
    if unknown.any():
        colors = np.unique(rgb[unknown].reshape(-1, 3), axis=0)[:8]
        raise ValueError("Unknown colors in {}: {}".format(path, colors.tolist()))
    return label


def sample_key(row: Mapping[str, str]) -> str:
    # The profiler prefixes the stable dataset name with a three-digit index.
    # The full string is identical across method exports when evaluation order
    # is unchanged, so retain it as the strict pairing key.
    return row["sample"]


def confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = (gt != IGNORE_LABEL) & (gt >= 0) & (gt < NUM_CLASSES)
    encoded = NUM_CLASSES * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
    matrix = matrix.astype(np.float64, copy=False)
    tp = np.diag(matrix)
    target = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = target + predicted - tp
    iou = np.divide(tp, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    miou = float(np.nanmean(iou))
    pixel_acc = float(tp.sum() / matrix.sum()) if matrix.sum() > 0 else math.nan
    return iou, miou, pixel_acc


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float) -> np.ndarray:
    """Official Boundary-IoU erosion rule for one binary mask."""
    binary = mask.astype(np.uint8)
    height, width = binary.shape
    dilation = max(1, int(round(dilation_ratio * math.sqrt(height ** 2 + width ** 2))))
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), dtype=np.uint8), iterations=dilation)
    eroded = eroded[1 : height + 1, 1 : width + 1]
    return binary - eroded


def load_export(export_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    csv_path = export_dir / "mask_inference" / "mask_inference_samples.csv"
    if not csv_path.is_file():
        raise FileNotFoundError("Missing full mask export: {}".format(csv_path))
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for row in read_csv(csv_path):
        key = sample_key(row)
        if key in pairs:
            raise ValueError("Duplicate sample {} in {}".format(key, csv_path))
        prediction = rgb_to_label(resolve(row["prediction"]))
        ground_truth = rgb_to_label(resolve(row["ground_truth"]))
        if prediction.shape != ground_truth.shape:
            prediction = cv2.resize(
                prediction, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        pairs[key] = (prediction, ground_truth)
    if not pairs:
        raise ValueError("No masks in {}".format(csv_path))
    return pairs


def ordered_confusions(pairs: Mapping[str, Tuple[np.ndarray, np.ndarray]], keys: Sequence[str]) -> np.ndarray:
    return np.stack([confusion(pairs[key][0], pairs[key][1]) for key in keys], axis=0)


def validate_pairing(exports: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]]) -> List[str]:
    reference_id = sorted(exports)[0]
    keys = sorted(exports[reference_id])
    reference = exports[reference_id]
    for analysis_id, pairs in exports.items():
        if sorted(pairs) != keys:
            missing = sorted(set(keys) - set(pairs))[:5]
            extra = sorted(set(pairs) - set(keys))[:5]
            raise ValueError("Pairing mismatch for {}: missing={}, extra={}".format(analysis_id, missing, extra))
        for key in keys:
            if not np.array_equal(reference[key][1], pairs[key][1]):
                raise ValueError("Ground-truth mismatch for {} sample {}".format(analysis_id, key))
    return keys


def boundary_iou(pairs: Mapping[str, Tuple[np.ndarray, np.ndarray]], ratio: float) -> np.ndarray:
    intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
    union = np.zeros(NUM_CLASSES, dtype=np.float64)
    for prediction, ground_truth in pairs.values():
        valid = ground_truth != IGNORE_LABEL
        for class_id in range(NUM_CLASSES):
            pred_boundary = mask_to_boundary((prediction == class_id) & valid, ratio).astype(bool)
            gt_boundary = mask_to_boundary((ground_truth == class_id) & valid, ratio).astype(bool)
            intersection[class_id] += np.logical_and(pred_boundary, gt_boundary).sum()
            union[class_id] += np.logical_or(pred_boundary, gt_boundary).sum()
    return np.divide(intersection, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)


def bootstrap_delta(
    remask: np.ndarray,
    baseline: np.ndarray,
    resamples: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    if remask.shape != baseline.shape:
        raise ValueError("Paired bootstrap arrays differ: {} vs {}".format(remask.shape, baseline.shape))
    rng = np.random.default_rng(seed)
    n_images = remask.shape[0]
    deltas = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 256):
        count = min(256, resamples - start)
        indices = rng.integers(0, n_images, size=(count, n_images))
        remask_matrices = remask[indices].sum(axis=1)
        baseline_matrices = baseline[indices].sum(axis=1)
        for offset in range(count):
            remask_miou = metrics_from_confusion(remask_matrices[offset])[1]
            baseline_miou = metrics_from_confusion(baseline_matrices[offset])[1]
            deltas[start + offset] = remask_miou - baseline_miou
    observed = metrics_from_confusion(remask.sum(axis=0))[1] - metrics_from_confusion(baseline.sum(axis=0))[1]
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    probability_positive = float(np.mean(deltas > 0))
    return float(observed), float(lower), float(upper), probability_positive


def transition_rows(
    setting: str,
    initial: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    final: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    keys: Sequence[str],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    initial_conf = ordered_confusions(initial, keys).sum(axis=0)
    final_conf = ordered_confusions(final, keys).sum(axis=0)
    _, initial_miou, initial_acc = metrics_from_confusion(initial_conf)
    _, final_miou, final_acc = metrics_from_confusion(final_conf)
    totals = np.zeros(NUM_CLASSES, dtype=np.float64)
    initial_correct_count = np.zeros(NUM_CLASSES, dtype=np.float64)
    initial_wrong_count = np.zeros(NUM_CLASSES, dtype=np.float64)
    corrected = np.zeros(NUM_CLASSES, dtype=np.float64)
    regressed = np.zeros(NUM_CLASSES, dtype=np.float64)
    changed = np.zeros(NUM_CLASSES, dtype=np.float64)
    for key in keys:
        initial_pred, ground_truth = initial[key]
        final_pred, final_ground_truth = final[key]
        if not np.array_equal(ground_truth, final_ground_truth):
            raise ValueError("Ground-truth mismatch in Stage-2 pair {}".format(key))
        valid = ground_truth != IGNORE_LABEL
        initial_correct = (initial_pred == ground_truth) & valid
        final_correct = (final_pred == ground_truth) & valid
        for class_id in range(NUM_CLASSES):
            class_mask = (ground_truth == class_id) & valid
            totals[class_id] += class_mask.sum()
            initial_correct_count[class_id] += np.logical_and(initial_correct, class_mask).sum()
            initial_wrong_count[class_id] += np.logical_and(~initial_correct, class_mask).sum()
            corrected[class_id] += np.logical_and.reduce((~initial_correct, final_correct, class_mask)).sum()
            regressed[class_id] += np.logical_and.reduce((initial_correct, ~final_correct, class_mask)).sum()
            changed[class_id] += np.logical_and(initial_pred != final_pred, class_mask).sum()

    def safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else math.nan

    overall = {
        "setting": setting,
        "comparison": "actual_stage2_initialization_to_selected_final",
        "n_images": len(keys),
        "stage2_init_miou": initial_miou,
        "final_miou": final_miou,
        "delta_miou": final_miou - initial_miou,
        "stage2_init_pixel_accuracy": initial_acc,
        "final_pixel_accuracy": final_acc,
        "delta_pixel_accuracy": final_acc - initial_acc,
        "corrected_fraction_of_initial_errors": safe_ratio(corrected.sum(), initial_wrong_count.sum()),
        "regressed_fraction_of_initial_correct": safe_ratio(regressed.sum(), initial_correct_count.sum()),
        "net_corrected_fraction_of_all_pixels": safe_ratio(corrected.sum() - regressed.sum(), totals.sum()),
        "prediction_changed_fraction": safe_ratio(changed.sum(), totals.sum()),
        "corrected_pixels": int(corrected.sum()),
        "regressed_pixels": int(regressed.sum()),
    }
    per_class: List[Dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "setting": setting,
                "class_id": class_id,
                "class_name": class_name,
                "pixels": int(totals[class_id]),
                "corrected_fraction_of_initial_errors": safe_ratio(corrected[class_id], initial_wrong_count[class_id]),
                "regressed_fraction_of_initial_correct": safe_ratio(
                    regressed[class_id], initial_correct_count[class_id]
                ),
                "net_corrected_fraction_of_class_pixels": safe_ratio(
                    corrected[class_id] - regressed[class_id], totals[class_id]
                ),
                "prediction_changed_fraction": safe_ratio(changed[class_id], totals[class_id]),
                "corrected_pixels": int(corrected[class_id]),
                "regressed_pixels": int(regressed[class_id]),
            }
        )
    return overall, per_class


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex_tables(
    output_dir: Path,
    boundary_rows: Sequence[Mapping[str, object]],
    bootstrap_rows: Sequence[Mapping[str, object]],
    transition_summary: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "% Auto-generated by scripts/analyze_paper_posthoc.py; inspect CSVs before use.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lcc}",
        r"\toprule Method & mIoU & Boundary IoU \\",
        r"\midrule",
    ]
    for setting in ("uiis_to_suim", "usis10k_to_suim"):
        title = "UIIS$\\rightarrow$SUIM" if setting.startswith("uiis") else "USIS10K$\\rightarrow$SUIM"
        lines.append(r"\multicolumn{3}{c}{" + title + r"} \\")
        for row in boundary_rows:
            if row["setting"] != setting:
                continue
            lines.append(
                "{} & {:.2f} & {:.2f} \\\\".format(
                    latex_escape(str(row["method_label"])),
                    100.0 * float(row["miou"]),
                    100.0 * float(row["mean_boundary_iou"]),
                )
            )
        if setting == "uiis_to_suim":
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Semantic Boundary IoU using a boundary width of 2\% of the image diagonal. Ordinary mIoU is repeated only as a result-lock check.}",
            r"\label{tab:boundary_iou}",
            r"\end{table}",
            "",
            "% Compact statistical comparison against the strongest baseline by locked mIoU.",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{lccc}",
            r"\toprule Setting & Baseline & $\Delta$mIoU & 95\% CI \\",
            r"\midrule",
        ]
    )
    for setting in ("uiis_to_suim", "usis10k_to_suim"):
        candidates = [row for row in bootstrap_rows if row["setting"] == setting]
        strongest = max(candidates, key=lambda row: float(row["baseline_miou"]))
        title = "UIIS$\\rightarrow$SUIM" if setting.startswith("uiis") else "USIS10K$\\rightarrow$SUIM"
        lines.append(
            "{} & {} & {:+.2f} & [{:+.2f}, {:+.2f}] \\\\".format(
                title,
                latex_escape(str(strongest["baseline_label"])),
                100.0 * float(strongest["delta_miou"]),
                100.0 * float(strongest["ci95_low"]),
                100.0 * float(strongest["ci95_high"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired image-level bootstrap comparison (10,000 resamples). The interval quantifies target-image sampling uncertainty, not training-seed variance.}",
            r"\label{tab:paired_bootstrap}",
            r"\end{table}",
            "",
            "% Stage-2 error-flow summary; the initial checkpoint is the actual continuation checkpoint.",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{lccccc}",
            r"\toprule Setting & Init. & Final & Corrected & Regressed & Net \\",
            r"\midrule",
        ]
    )
    for row in transition_summary:
        title = "UIIS$\\rightarrow$SUIM" if str(row["setting"]).startswith("uiis") else "USIS10K$\\rightarrow$SUIM"
        lines.append(
            "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:+.2f} \\\\".format(
                title,
                100.0 * float(row["stage2_init_miou"]),
                100.0 * float(row["final_miou"]),
                100.0 * float(row["corrected_fraction_of_initial_errors"]),
                100.0 * float(row["regressed_fraction_of_initial_correct"]),
                100.0 * float(row["net_corrected_fraction_of_all_pixels"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Stage-2 error transition. Corrected is the fraction of initially wrong pixels made correct; Regressed is the fraction of initially correct pixels made wrong; Net is normalized by all valid pixels.}",
            r"\label{tab:stage2_transition}",
            r"\end{table}",
        ]
    )
    (output_dir / "candidate_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.boundary_ratio <= 0:
        raise ValueError("--boundary-ratio must be positive")
    if args.bootstrap_resamples < 100:
        raise ValueError("--bootstrap-resamples must be at least 100")
    manifest = read_csv(resolve(args.manifest))
    export_root = resolve(args.export_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exports: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for row in manifest:
        exports[row["analysis_id"]] = load_export(export_root / row["analysis_id"])
    keys = validate_pairing(exports)

    integrity_rows: List[Dict[str, object]] = []
    confusion_by_id: Dict[str, np.ndarray] = {}
    for row in manifest:
        analysis_id = row["analysis_id"]
        matrices = ordered_confusions(exports[analysis_id], keys)
        confusion_by_id[analysis_id] = matrices
        iou, observed, pixel_acc = metrics_from_confusion(matrices.sum(axis=0))
        expected = float(row["expected_miou"])
        difference = observed - expected
        status = "ok" if abs(difference) <= args.miou_tolerance else "mismatch"
        integrity_rows.append(
            {
                "analysis_id": analysis_id,
                "setting": row["setting"],
                "method": row["method"],
                "role": row["role"],
                "n_images": len(keys),
                "expected_miou": expected,
                "observed_miou": observed,
                "difference": difference,
                "pixel_accuracy": pixel_acc,
                "status": status,
                **{"{}_iou".format(CLASS_NAMES[idx].lower()): iou[idx] for idx in range(NUM_CLASSES)},
            }
        )
    write_csv(output_dir / "integrity_checks.csv", integrity_rows)
    mismatches = [row for row in integrity_rows if row["status"] != "ok"]
    if mismatches:
        raise RuntimeError(
            "Result-lock mismatch; refusing derived analyses. Inspect {}. First mismatch: {}".format(
                output_dir / "integrity_checks.csv", mismatches[0]
            )
        )

    main_rows = [row for row in manifest if row["role"] == "main"]
    boundary_rows: List[Dict[str, object]] = []
    for row in main_rows:
        analysis_id = row["analysis_id"]
        iou, miou, pixel_acc = metrics_from_confusion(confusion_by_id[analysis_id].sum(axis=0))
        biou = boundary_iou(exports[analysis_id], args.boundary_ratio)
        boundary_rows.append(
            {
                "setting": row["setting"],
                "method": row["method"],
                "method_label": row["method_label"],
                "n_images": len(keys),
                "miou": miou,
                "pixel_accuracy": pixel_acc,
                "mean_boundary_iou": float(np.nanmean(biou)),
                "foreground_boundary_iou": float(np.nanmean(biou[1:])),
                **{"{}_boundary_iou".format(CLASS_NAMES[idx].lower()): biou[idx] for idx in range(NUM_CLASSES)},
            }
        )
    boundary_rows.sort(key=lambda row: (str(row["setting"]), -float(row["mean_boundary_iou"])))
    write_csv(output_dir / "boundary_iou.csv", boundary_rows)

    bootstrap_rows: List[Dict[str, object]] = []
    for setting in sorted({row["setting"] for row in main_rows}):
        setting_rows = [row for row in main_rows if row["setting"] == setting]
        remask_row = next(row for row in setting_rows if row["method"] == "remask")
        remask_matrices = confusion_by_id[remask_row["analysis_id"]]
        remask_miou = metrics_from_confusion(remask_matrices.sum(axis=0))[1]
        for baseline_row in setting_rows:
            if baseline_row["method"] == "remask":
                continue
            baseline_matrices = confusion_by_id[baseline_row["analysis_id"]]
            baseline_miou = metrics_from_confusion(baseline_matrices.sum(axis=0))[1]
            observed, lower, upper, probability_positive = bootstrap_delta(
                remask_matrices,
                baseline_matrices,
                args.bootstrap_resamples,
                args.bootstrap_seed,
            )
            bootstrap_rows.append(
                {
                    "setting": setting,
                    "baseline": baseline_row["method"],
                    "baseline_label": baseline_row["method_label"],
                    "n_images": len(keys),
                    "resamples": args.bootstrap_resamples,
                    "seed": args.bootstrap_seed,
                    "remask_miou": remask_miou,
                    "baseline_miou": baseline_miou,
                    "delta_miou": observed,
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "bootstrap_probability_delta_positive": probability_positive,
                }
            )
    bootstrap_rows.sort(key=lambda row: (str(row["setting"]), -float(row["baseline_miou"])))
    write_csv(output_dir / "paired_bootstrap_miou.csv", bootstrap_rows)

    transition_summary: List[Dict[str, object]] = []
    transition_classes: List[Dict[str, object]] = []
    for setting in sorted({row["setting"] for row in main_rows}):
        initial_row = next(row for row in manifest if row["setting"] == setting and row["role"] == "stage2_init")
        final_row = next(
            row for row in manifest if row["setting"] == setting and row["role"] == "main" and row["method"] == "remask"
        )
        summary, per_class = transition_rows(
            setting,
            exports[initial_row["analysis_id"]],
            exports[final_row["analysis_id"]],
            keys,
        )
        summary["stage2_init_checkpoint"] = initial_row["checkpoint"]
        summary["final_checkpoint"] = final_row["checkpoint"]
        transition_summary.append(summary)
        transition_classes.extend(per_class)
    write_csv(output_dir / "stage2_transition_summary.csv", transition_summary)
    write_csv(output_dir / "stage2_transition_per_class.csv", transition_classes)

    protocol = {
        "ordinary_segmentation": "Dataset-level confusion matrix; per-class IoU; unweighted mean over classes; ignored pixels excluded.",
        "boundary_iou": {
            "metric": "Boundary IoU (Cheng et al., CVPR 2021), applied independently to each semantic class",
            "dilation_ratio": args.boundary_ratio,
            "boundary_width": "round(dilation_ratio * image diagonal), minimum 1 pixel",
            "aggregation": "boundary intersections and unions accumulated over the test split, then unweighted mean over classes",
        },
        "bootstrap": {
            "unit": "target test image",
            "paired": True,
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "interval": "two-sided 95% percentile interval",
            "scope": "target-image sampling uncertainty; not training-seed variance",
        },
        "stage2_transition": {
            "comparison": "actual checkpoint used to initialize Stage 2 versus selected final ReMask checkpoint",
            "corrected": "Stage-2-init wrong and final correct",
            "regressed": "Stage-2-init correct and final wrong",
            "warning": "stage1_reported rows are checked for provenance but are not used as transition starts when they differ from the actual continuation checkpoint",
        },
        "result_lock": {
            "manifest": str(resolve(args.manifest)),
            "tolerance": args.miou_tolerance,
            "n_images": len(keys),
        },
    }
    with (output_dir / "analysis_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    write_latex_tables(output_dir, boundary_rows, bootstrap_rows, transition_summary)

    print("All {} checkpoint exports passed the result lock on {} images.".format(len(manifest), len(keys)))
    print("Boundary IoU: {}".format(output_dir / "boundary_iou.csv"))
    print("Paired bootstrap: {}".format(output_dir / "paired_bootstrap_miou.csv"))
    print("Stage-2 transitions: {}".format(output_dir / "stage2_transition_summary.csv"))
    print("Candidate LaTeX tables: {}".format(output_dir / "candidate_tables.tex"))


if __name__ == "__main__":
    main()
