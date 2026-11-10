"""Compare a completed source-only evaluation with DGLE Tables 1 and 2.

The reference numbers are published scores, not acceptance thresholds. A close
score alone cannot prove identical training or preprocessing because DGLE does
not release all source-training and evaluation details.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional


REFERENCE = {
    "vh2pd": {
        "split": "isprs_pd_rgb_eval",
        "classes": ["impervious_surface", "building", "low_vegetation", "tree", "car", "clutter"],
        "iou_percent": [45.49, 53.15, 21.60, 3.21, 74.33, 1.33],
        "miou_percent": 33.18,
        "miou5_percent": 39.55,
        "table": "Table 1",
    },
    "r2u": {
        "split": "loveda_urban_val",
        "classes": ["background", "building", "road", "water", "barren", "forest", "agriculture"],
        "iou_percent": [32.89, 26.80, 28.24, 55.20, 20.20, 28.73, 44.92],
        "miou_percent": 33.86,
        "table": "Table 2",
    },
}
PAPER_URL = "https://arxiv.org/html/2509.18502"


def compare(evaluation: dict, protocol: Optional[dict] = None) -> dict:
    task = evaluation["task"]
    if task not in REFERENCE:
        raise ValueError(f"No DGLE source-only reference for task {task!r}")
    reference = REFERENCE[task]
    if evaluation.get("split") != reference["split"]:
        raise ValueError("Evaluation split does not match DGLE target evaluation split")
    metrics = evaluation["metrics"]
    observed_iou = metrics["iou"]
    if len(observed_iou) != len(reference["classes"]) or any(value is None for value in observed_iou):
        raise ValueError("All DGLE classes must have a defined IoU")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in observed_iou):
        raise ValueError("Observed IoU must be finite fractions in [0, 1]")
    observed_mean = sum(observed_iou) / len(observed_iou)
    if not math.isclose(metrics["miou"], observed_mean, abs_tol=1e-6):
        raise ValueError("Reported mIoU is inconsistent with per-class IoU")
    rows = []
    for name, published, observed in zip(reference["classes"], reference["iou_percent"], observed_iou):
        actual = 100 * observed
        rows.append({"class": name, "published_percent": published,
                     "observed_percent": round(actual, 4),
                     "delta_percentage_points": round(actual - published, 4)})
    scores = {"miou": {"published_percent": reference["miou_percent"],
                       "observed_percent": round(100 * metrics["miou"], 4),
                       "delta_percentage_points": round(100 * metrics["miou"] - reference["miou_percent"], 4)}}
    if task == "vh2pd":
        observed_miou5 = sum(observed_iou[:5]) / 5
        if not math.isclose(metrics["miou5"], observed_miou5, abs_tol=1e-6):
            raise ValueError("Reported mIoU5 is inconsistent with per-class IoU")
        scores["miou5"] = {"published_percent": reference["miou5_percent"],
                           "observed_percent": round(100 * metrics["miou5"], 4),
                           "delta_percentage_points": round(100 * metrics["miou5"] - reference["miou5_percent"], 4)}
    if protocol is not None:
        if protocol.get("stage") != "eval" or protocol.get("task") != task or protocol.get("split") != reference["split"]:
            raise ValueError("Evaluation protocol does not match metrics task/split")
        files = protocol["files_sha256"]
        if files["source_checkpoint"] != evaluation["source_checkpoint_sha256"]:
            raise ValueError("Source checkpoint hash differs between protocol and metrics")
    else:
        files = {}
    return {
        "task": task, "split": reference["split"], "paper": PAPER_URL,
        "table": reference["table"], "reference_row": "Source-Only",
        "source_checkpoint_sha256": evaluation["source_checkpoint_sha256"],
        "manifest_sha256": files.get("manifest"),
        "dataset_metadata_sha256": files.get("dataset_metadata"),
        "valid_pixels": metrics["valid_pixels"], "scores": scores, "classes": rows,
        "interpretation": "Numerical comparison only; DGLE does not specify every source-training and evaluation detail.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_eval", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol_path = args.metrics_eval.with_name("protocol_eval.json")
    report = compare(json.loads(args.metrics_eval.read_text(encoding="utf-8")),
                     json.loads(protocol_path.read_text(encoding="utf-8")))
    output = args.output or args.metrics_eval.with_name("dgle_source_only_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DGLE {report['table']} Source-Only comparison: {report['task']}")
    for name, item in report["scores"].items():
        print(f"  {name}: observed {item['observed_percent']:.2f}%, "
              f"published {item['published_percent']:.2f}%, "
              f"delta {item['delta_percentage_points']:+.2f} pp")
    print(f"Per-class results and provenance: {output}")


if __name__ == "__main__":
    main()
