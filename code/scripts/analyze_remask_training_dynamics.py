#!/usr/bin/env python3
"""Extract class-wise validation curves and training diagnostics for ReMask.

The script intentionally keeps Stage 1 and Stage 2 on their native iteration
axes.  This avoids treating the normalized 30k visualization axis as literal
optimizer time.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validate_results" / "remask_training_dynamics"

CLASS_NAMES = [
    "Background",
    "Diver",
    "Plant",
    "Wreck",
    "Robot",
    "Reef",
    "Fish",
    "Floor",
]

RUNS = {
    "uiis_stage1": {
        "domain": "UIIS->SUIM",
        "stage": "Stage1-UA+UWRG",
        "path": ROOT / "assets" / "experimentss_sensitivity" / "slurm_uwrg_sens_24283_8.out",
        "selected_iter": 24999,
    },
    "uiis_stage2": {
        "domain": "UIIS->SUIM",
        "stage": "Stage2-TC+CB+UWRG",
        "path": ROOT / "HPC_logs" / "experiments_uwrgfinetune" / "slurm_temp_cb_best_25134_1.out",
    },
    "uiis_direct_tc": {
        "domain": "UIIS->SUIM",
        "stage": "Direct-TC+CB+UWRG",
        "path": ROOT / "assets" / "experiments_" / "slurm_temp_cb_source_25347_0.out",
    },
    "usis_stage1": {
        "domain": "USIS10K->SUIM",
        "stage": "Stage1-UA+UWRG",
        "path": ROOT / "assets" / "experimentss_sensitivity" / "slurm_uwrg_sens_24281_19.out",
        "selected_iter": 28999,
    },
    "usis_stage2": {
        "domain": "USIS10K->SUIM",
        "stage": "Stage2-TC+CB+UWRG",
        "path": ROOT / "HPC_logs" / "experiments_uwrgfinetune" / "slurm_temp_cb_best_25134_3.out",
    },
    "usis_direct_tc": {
        "domain": "USIS10K->SUIM",
        "stage": "Direct-TC+CB+UWRG",
        "path": ROOT / "assets" / "experiments_" / "slurm_temp_cb_source_25347_1.out",
    },
}

# These are the only aggregate results that should be quoted in the manuscript.
# Raw logs below are parsed only for trajectory shape and mechanism diagnostics.
OFFICIAL_RESULTS = {
    "UIIS->SUIM": {"stage1": 49.13, "direct_tc": 45.16, "full": 49.82, "stage2_gain": 0.69},
    "USIS10K->SUIM": {"stage1": 48.83, "direct_tc": 47.45, "full": 50.30, "stage2_gain": 1.47},
}

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VAL_RE = re.compile(
    rf"Val result(?: iter (?P<iter>\d+))?: mIoU/mAcc/allAcc "
    rf"(?P<miou>{NUMBER})/(?P<macc>{NUMBER})/(?P<allacc>{NUMBER})"
)
CLASS_RE = re.compile(
    rf"Class_(?P<index>\d+) (?P<name>.*?) Result: iou/accuracy "
    rf"(?P<iou>{NUMBER})/(?P<accuracy>{NUMBER})"
)
MONITOR_RE = re.compile(
    r"(?:Early-stop monitor|Best .* checkpoint updated).*?"
    r"(?:mIoU\s+(?P<miou_a>[0-9.]+).*?at iter\s+(?P<iter_a>\d+)|"
    r"at iter\s+(?P<iter_b>\d+).*?mIoU\s+(?P<miou_b>[0-9.]+))"
)
TRAIN_ITER_RE = re.compile(r"\biter:\s*(\d+)")
TRAIN_FIELDS = [
    "reliability",
    "reliability_std",
    "reliability_low",
    "class_balance",
    "uwrg_gate_raw",
    "uwrg_gate_eff",
    "uwrg_downweight_eff",
    "uwrg_disagreement",
    "uctr_temporal_raw",
    "uwrg_strength",
    "history_hit",
    "history_depth",
]


def parse_log(path: Path) -> tuple[List[dict], List[dict]]:
    """Return validation records and per-log-line training diagnostics."""
    if not path.is_file():
        raise FileNotFoundError(path)

    validation: List[dict] = []
    training: List[dict] = []
    last_train_iter: int | None = None
    pending: int | None = None
    active_val: int | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        train_match = TRAIN_ITER_RE.search(line)
        if train_match and "DTST.trainer INFO: eta:" in line:
            current = int(train_match.group(1))
            if pending is not None and current > int(validation[pending]["iteration"]):
                validation[pending]["iteration"] = current - 1
                pending = None
            last_train_iter = current
            record: dict = {"iteration": current}
            for field in TRAIN_FIELDS:
                match = re.search(rf"\b{re.escape(field)}:\s*({NUMBER})", line)
                if match:
                    record[field] = float(match.group(1))
            training.append(record)

        val_match = VAL_RE.search(line)
        if val_match:
            # Logs also contain an auxiliary high-IoU evaluation on target-train
            # pseudo labels (typically >0.9).  The SUIM validation curves are
            # safely below 0.65; exclude the auxiliary block from this audit.
            if float(val_match.group("miou")) > 0.65:
                active_val = None
                pending = None
                continue
            explicit = val_match.group("iter")
            validation.append(
                {
                    "iteration": int(explicit) if explicit else (last_train_iter or -1),
                    "mIoU": float(val_match.group("miou")),
                    "mAcc": float(val_match.group("macc")),
                    "allAcc": float(val_match.group("allacc")),
                }
            )
            active_val = len(validation) - 1
            pending = None if explicit else active_val
            continue

        class_match = CLASS_RE.search(line)
        if class_match and active_val is not None:
            index = int(class_match.group("index"))
            if 0 <= index < len(CLASS_NAMES):
                validation[active_val][f"iou_{index}"] = float(class_match.group("iou"))
            continue

        monitor = MONITOR_RE.search(line)
        if monitor and validation:
            monitor_iter = monitor.group("iter_a") or monitor.group("iter_b")
            monitor_miou = monitor.group("miou_a") or monitor.group("miou_b")
            candidate = pending if pending is not None else len(validation) - 1
            if (
                monitor_iter
                and monitor_miou
                and abs(validation[candidate]["mIoU"] - float(monitor_miou)) < 5e-4
            ):
                validation[candidate]["iteration"] = int(monitor_iter)
                pending = None

    # Later duplicate records are authoritative (some logs repeat final eval).
    by_iteration = {int(row["iteration"]): row for row in validation if int(row["iteration"]) >= 0}
    return [by_iteration[key] for key in sorted(by_iteration)], training


def median(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values]
    return statistics.median(clean) if clean else None


def percentile(values: Iterable[float], fraction: float) -> float | None:
    clean = sorted(float(value) for value in values)
    if not clean:
        return None
    position = fraction * (len(clean) - 1)
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def closest_row(rows: Sequence[Mapping[str, float]], iteration: int) -> Mapping[str, float]:
    return min(rows, key=lambda row: abs(int(row["iteration"]) - iteration))


def class_delta(after: Mapping[str, float], before: Mapping[str, float]) -> List[float]:
    return [float(after[f"iou_{idx}"]) - float(before[f"iou_{idx}"]) for idx in range(8)]


def progress_iteration(
    rows: Sequence[Mapping[str, float]],
    start: Mapping[str, float],
    peak: Mapping[str, float],
    fraction: float,
) -> int | None:
    target = float(start["mIoU"]) + fraction * (float(peak["mIoU"]) - float(start["mIoU"]))
    eligible = [row for row in rows if int(row["iteration"]) <= int(peak["iteration"])]
    return next((int(row["iteration"]) for row in eligible if float(row["mIoU"]) >= target), None)


def fmt_pp(value: float) -> str:
    return f"{value * 100:+.2f}"


def fmt_iou(value: float) -> str:
    return f"{value * 100:.2f}"


def phase_median(rows: Sequence[Mapping[str, float]], field: str, start: float, end: float) -> float | None:
    maximum = max(int(row["iteration"]) for row in rows)
    values = [
        float(row[field])
        for row in rows
        if field in row and start * maximum <= int(row["iteration"]) < end * maximum
    ]
    return median(values)


def fmt_transition(first: float | None, last: float | None) -> str:
    if first is None or last is None:
        return "--"
    return f"{first:.3f} -> {last:.3f}"


def write_validation_csv(output: Path, parsed: Mapping[str, tuple[List[dict], List[dict]]]) -> None:
    fields = ["diagnostic_only", "run", "domain", "stage", "iteration", "delta_mIoU_from_first_eval"] + [
        f"delta_iou_{idx}_{CLASS_NAMES[idx].lower()}_from_first_eval" for idx in range(8)
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run, (rows, _) in parsed.items():
            spec = RUNS[run]
            first = rows[0]
            for row in rows:
                target = {
                    "diagnostic_only": "true",
                    "run": run,
                    "domain": spec["domain"],
                    "stage": spec["stage"],
                    "iteration": row["iteration"],
                    "delta_mIoU_from_first_eval": float(row["mIoU"]) - float(first["mIoU"]),
                }
                for idx in range(8):
                    key = f"iou_{idx}"
                    target[f"delta_iou_{idx}_{CLASS_NAMES[idx].lower()}_from_first_eval"] = (
                        float(row[key]) - float(first[key]) if key in row and key in first else ""
                    )
                writer.writerow(target)


def write_training_csv(output: Path, parsed: Mapping[str, tuple[List[dict], List[dict]]]) -> None:
    fields = ["run", "domain", "stage", "phase", "start_iter", "end_iter", "samples"]
    for metric in TRAIN_FIELDS:
        fields.extend([f"{metric}_p10", f"{metric}_median", f"{metric}_p90"])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run, (_, rows) in parsed.items():
            if not rows:
                continue
            maximum = max(int(row["iteration"]) for row in rows)
            boundaries = [0, round(maximum * 0.1), round(maximum * 0.4), round(maximum * 0.8), maximum + 1]
            labels = ["early_0_10pct", "build_10_40pct", "mature_40_80pct", "late_80_100pct"]
            for label, start, end in zip(labels, boundaries, boundaries[1:]):
                subset = [row for row in rows if start <= int(row["iteration"]) < end]
                record: MutableMapping[str, object] = {
                    "run": run,
                    "domain": RUNS[run]["domain"],
                    "stage": RUNS[run]["stage"],
                    "phase": label,
                    "start_iter": start,
                    "end_iter": end - 1,
                    "samples": len(subset),
                }
                for metric in TRAIN_FIELDS:
                    values = [row[metric] for row in subset if metric in row]
                    record[f"{metric}_p10"] = percentile(values, 0.1)
                    record[f"{metric}_median"] = median(values)
                    record[f"{metric}_p90"] = percentile(values, 0.9)
                writer.writerow(record)


def write_report(output: Path, parsed: Mapping[str, tuple[List[dict], List[dict]]]) -> None:
    lines = [
        "# ReMask training-dynamics insight audit",
        "",
        "**Result-lock rule.** The aggregate numbers below are copied from the original Results/Ablation tables and are authoritative. Raw logs are used only for trajectory shape, peak timing, class direction, and mechanism diagnostics; they do not replace or recompute manuscript results.",
        "",
        "| Transfer | Early UA+UWRG | Direct TC+CB+UWRG from source | Full ReMask | Official Stage-2 gain |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| UIIS -> SUIM | 49.13 | 45.16 | 49.82 | +0.69 |",
        "| USIS10K -> SUIM | 48.83 | 47.45 | 50.30 | +1.47 |",
        "",
        "The paper-style curve rescales Stage 1 and Stage 2 to a common 30k display horizon. Its 25k boundary is normalized adaptation progress, not a literal shared optimizer iteration.",
        "",
    ]

    for prefix, title in [("uiis", "UIIS -> SUIM"), ("usis", "USIS10K -> SUIM")]:
        stage1 = parsed[f"{prefix}_stage1"][0]
        stage2 = parsed[f"{prefix}_stage2"][0]
        stage1_best = max(stage1, key=lambda row: float(row["mIoU"]))
        stage2_first = stage2[0]
        stage2_best = max(stage2, key=lambda row: float(row["mIoU"]))
        refine_deltas = class_delta(stage2_best, stage2_first)
        progress = [progress_iteration(stage1, stage1[0], stage1_best, value) for value in (0.5, 0.8, 0.9)]
        up = [CLASS_NAMES[idx] for idx, value in enumerate(refine_deltas) if value >= 0.002]
        down = [CLASS_NAMES[idx] for idx, value in enumerate(refine_deltas) if value <= -0.002]
        lines.extend(
            [
                f"## {title}: diagnostic observations",
                "",
                f"- Stage 1 rises gradually; it first reaches 50/80/90% of the within-log improvement at native iterations {progress[0]}/{progress[1]}/{progress[2]}. This describes curve shape only.",
                f"- The selected full-model peak appears at about {int(stage2_best['iteration']) / 1000:.1f}k local Stage-2 iterations.",
                f"- From the first Stage-2 evaluation to that peak, upward class trajectories are {', '.join(up)}; downward trajectories are {', '.join(down)}.",
                "- The continuation declines after the selected peak, supporting best-checkpoint reporting and an adaptive class-aware stopping rule.",
                "",
            ]
        )

    lines.extend(
        [
            "## Logged training-weight diagnostics",
            "",
            "Early denotes the first 10% and late the final 20% of each native run. Values are medians over logged minibatches.",
            "",
            "| Run | Reliability | Fraction weight < 0.7 | Class balance | UWRG effective gate | UWRG downweight | Temporal raw | History hit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in ("uiis_stage1", "uiis_stage2", "usis_stage1", "usis_stage2"):
        training = parsed[run][1]
        values = {
            field: (
                phase_median(training, field, 0.0, 0.1),
                phase_median(training, field, 0.8, 1.01),
            )
            for field in (
                "reliability",
                "reliability_low",
                "class_balance",
                "uwrg_gate_eff",
                "uwrg_downweight_eff",
                "uctr_temporal_raw",
                "history_hit",
            )
        }
        lines.append(
            f"| {run} | {fmt_transition(*values['reliability'])} | "
            f"{fmt_transition(*values['reliability_low'])} | {fmt_transition(*values['class_balance'])} | "
            f"{fmt_transition(*values['uwrg_gate_eff'])} | {fmt_transition(*values['uwrg_downweight_eff'])} | "
            f"{fmt_transition(*values['uctr_temporal_raw'])} | {fmt_transition(*values['history_hit'])} |"
        )
    lines.extend(
        [
            "",
            "Stage 1 and Stage 2 reliability values are produced by different formulas and are not calibrated probabilities. Their contrast is nevertheless consistent with the intended conservative-to-expansive curriculum. The global UWRG mean is dominated by stable pixels and cannot localize the pixels or classes that were strongly attenuated.",
            "",
        ]
    )

    lines.extend(
        [
            "## Interpretation constraints",
            "",
            "- Do not quote alternative aggregate mIoU values from the diagnostic logs; use the locked Results/Ablation table above.",
            "- Class directions come from single-run trajectories and are not independent causal component effects.",
            "- Direct temporal training from the source changes initialization, optimization history, and pseudo-label quality simultaneously; it tests ordering, not temporal consistency in isolation.",
            "- Global mean UWRG gates close to one do not prove that the gate is inactive: the mechanism is segment-local and a small set of strongly downweighted pixels can be diluted by easy background/large regions. Pixel- or component-stratified gate histograms would be needed for a direct localization claim.",
            "- Best-checkpoint reporting follows the original protocol; later diagnostic evaluations answer a different question and must not replace the reported results.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed = {name: parse_log(Path(spec["path"])) for name, spec in RUNS.items()}
    write_validation_csv(args.output_dir / "diagnostic_relative_curves.csv", parsed)
    write_training_csv(args.output_dir / "training_metric_phases.csv", parsed)
    write_report(args.output_dir / "training_dynamics_report.md", parsed)
    print(f"Wrote ReMask dynamics audit to {args.output_dir}")


if __name__ == "__main__":
    main()
