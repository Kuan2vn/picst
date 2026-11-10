#!/usr/bin/env python3
"""Export full SUIM predictions for one checkpoint in the post-hoc manifest.

This is an evaluation-only wrapper around profile_uiis_to_suim_efficiency.py.
It never trains or modifies a checkpoint.  After inference it rejects an export
whose recomputed mIoU does not match the result-locked expected value.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="scripts/paper_posthoc_checkpoints.csv")
    parser.add_argument("--index", type=int, required=True, help="Zero-based manifest row index.")
    parser.add_argument("--output-root", default="validate_results/paper_posthoc_exports")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", default="960x480")
    parser.add_argument("--miou-tolerance", type=float, default=7.5e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_row(path: Path, index: int) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError("Manifest index {} outside [0, {}).".format(index, len(rows)))
    return rows[index]


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    row = load_row(manifest_path, args.index)
    checkpoint = resolve(row["checkpoint"])
    source_checkpoint = resolve(row["source_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing analysis checkpoint: {}".format(checkpoint))
    if not source_checkpoint.is_file():
        raise FileNotFoundError("Missing source checkpoint: {}".format(source_checkpoint))

    output_dir = resolve(args.output_root) / row["analysis_id"]
    result_path = output_dir / "extra_analysis.json"
    if result_path.is_file() and not args.overwrite:
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        observed = float(result["overall_miou"])
        expected = float(row["expected_miou"])
        if abs(observed - expected) <= args.miou_tolerance:
            print("Verified existing export {}: mIoU {:.6f}".format(row["analysis_id"], observed))
            return
        raise RuntimeError(
            "Existing export mIoU {:.6f} does not match expected {:.6f}; "
            "rerun with --overwrite only after checking provenance.".format(observed, expected)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    profile = ROOT / "scripts" / "profile_uiis_to_suim_efficiency.py"
    command: List[str] = [
        args.python,
        str(profile),
        "--method",
        row["method"],
        "--python",
        args.python,
        "--config-file",
        row["config_file"],
        "--source-checkpoint",
        str(source_checkpoint),
        "--source-train",
        row["source_train"],
        "--target-train",
        "suim_train",
        "--test-set",
        "suim_val",
        "--tt-set",
        "suim_TT_train",
        "--ii-set",
        "suim_II_train",
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        "4",
        "--profile-batches",
        "1",
        "--skip-training-profile",
        "--input-size",
        args.input_size,
        "--latency-warmup",
        "1",
        "--latency-repeats",
        "1",
        "--analysis-run-dir",
        str(checkpoint.parent),
        "--analysis-checkpoint",
        str(checkpoint),
        "--analysis-max-batches",
        "0",
        "--analysis-batch-size",
        "1",
        "--mask-samples",
        "1000000",
        "--extra-analysis",
        "--strict-extra-analysis",
    ]
    print("Exporting {} ({}, {})".format(row["analysis_id"], row["setting"], row["role"]))
    print("Checkpoint: {}".format(checkpoint))
    subprocess.run(command, cwd=str(ROOT), check=True)

    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    observed = float(result["overall_miou"])
    expected = float(row["expected_miou"])
    delta = observed - expected
    provenance = {
        "manifest": str(manifest_path),
        "manifest_index": args.index,
        "manifest_row": row,
        "observed_miou": observed,
        "expected_miou": expected,
        "absolute_difference": abs(delta),
        "tolerance": args.miou_tolerance,
        "integrity_status": "ok" if abs(delta) <= args.miou_tolerance else "mismatch",
        "command": command,
    }
    with (output_dir / "posthoc_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    if abs(delta) > args.miou_tolerance:
        raise RuntimeError(
            "Result-lock failure for {}: observed {:.6f}, expected {:.6f}, delta {:+.6f}. "
            "Do not use this export.".format(row["analysis_id"], observed, expected, delta)
        )
    print("Result lock passed: observed {:.6f}, expected {:.6f}".format(observed, expected))


if __name__ == "__main__":
    main()
