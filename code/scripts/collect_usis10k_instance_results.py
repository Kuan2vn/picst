#!/usr/bin/env python
"""Collect USIS10K instance-segmentation runs into comparison CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


METHOD_ORDER = {
    "base": 0,
    "base_labeled": 0,
    "st": 1,
    "static_confmatch": 2,
    "fixmatch": 3,
    "confmatch": 4,
    "remask": 5,
    "remask_cb": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Directory containing method run folders, e.g. outputs/.../multi/mask_rcnn.",
    )
    parser.add_argument("--out-prefix", default="instance_results")
    return parser.parse_args()


def method_key(name: str) -> tuple[int, str]:
    base = re.sub(r"_r\d.*$", "", name)
    for prefix, order in METHOD_ORDER.items():
        if base == prefix or base.startswith(prefix + "_"):
            return order, name
    return 99, name


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def finite_values(records: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def add_stats(row: Dict[str, Any], records: List[Dict[str, Any]], key: str) -> None:
    values = finite_values(records, key)
    if not values:
        return
    prefix = key.replace("/", "_")
    row[f"{prefix}_min"] = min(values)
    row[f"{prefix}_mean"] = sum(values) / len(values)
    row[f"{prefix}_max"] = max(values)
    row[f"{prefix}_first"] = values[0]
    row[f"{prefix}_last"] = values[-1]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect_run(run_dir: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    records = read_jsonl(run_dir / "metrics.json")
    eval_records = [record for record in records if "segm/AP" in record]
    train_records = [record for record in records if "iteration" in record and "total_loss" in record]

    summary: Dict[str, Any] = {
        "method": run_dir.name,
        "train_rows": len(train_records),
        "eval_rows": len(eval_records),
    }
    eval_rows: List[Dict[str, Any]] = []
    if eval_records:
        best = max(eval_records, key=lambda record: float(record.get("segm/AP", float("-inf"))))
        final = eval_records[-1]
        summary.update(
            {
                "best_iteration": best.get("iteration"),
                "final_iteration": final.get("iteration"),
            }
        )
        for key in [
            "segm/AP",
            "segm/AP50",
            "segm/AP75",
            "segm/APs",
            "segm/APm",
            "segm/APl",
            "bbox/AP",
            "bbox/AP50",
            "bbox/AP75",
        ]:
            summary["best_" + key.replace("/", "_")] = best.get(key)
            summary["final_" + key.replace("/", "_")] = final.get(key)

        for record in eval_records:
            row = {"method": run_dir.name, "iteration": record.get("iteration")}
            for key in [
                "segm/AP",
                "segm/AP50",
                "segm/AP75",
                "segm/APs",
                "segm/APm",
                "segm/APl",
                "bbox/AP",
                "bbox/AP50",
                "bbox/AP75",
            ]:
                row[key] = record.get(key)
            eval_rows.append(row)

    diagnostics: Dict[str, Any] = {
        "method": run_dir.name,
        "train_rows": len(train_records),
        "eval_rows": len(eval_records),
    }
    for key in [
        "total_loss",
        "loss_cls",
        "loss_box_reg",
        "loss_mask",
        "loss_rpn_cls",
        "loss_rpn_loc",
        "online/pseudo_instances",
        "online/reliability",
        "online/reliability_min",
        "online/reliability_max",
        "online/supervised_images",
        "online/unlabeled_images",
        "online/memory_size",
        "lr",
    ]:
        add_stats(diagnostics, train_records, key)

    pred_path = run_dir / "inference" / "coco_instances_results.json"
    if pred_path.exists():
        try:
            predictions = json.loads(pred_path.read_text(errors="ignore"))
        except json.JSONDecodeError:
            predictions = []
        diagnostics["final_prediction_count"] = len(predictions)
        if predictions:
            scores = [float(item.get("score", 0.0)) for item in predictions]
            diagnostics["final_score_mean"] = sum(scores) / len(scores)
            diagnostics["final_score_max"] = max(scores)
            diagnostics["final_images_with_predictions"] = len(
                Counter(item.get("image_id") for item in predictions)
            )
            diagnostics["final_predictions_score_ge_0_5"] = sum(score >= 0.5 for score in scores)

    return summary, eval_rows, diagnostics


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    run_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / "metrics.json").exists()],
        key=lambda path: method_key(path.name),
    )
    summaries: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        summary, per_eval, diagnostic = collect_run(run_dir)
        summaries.append(summary)
        eval_rows.extend(per_eval)
        diagnostics.append(diagnostic)

    write_csv(root / f"{args.out_prefix}_best.csv", summaries)
    write_csv(root / f"{args.out_prefix}_eval_history.csv", eval_rows)
    write_csv(root / f"{args.out_prefix}_diagnostics.csv", diagnostics)

    for row in summaries:
        print(
            f"{row['method']}: "
            f"best_segm_AP={row.get('best_segm_AP', 'NA')} "
            f"final_segm_AP={row.get('final_segm_AP', 'NA')}"
        )


if __name__ == "__main__":
    main()
