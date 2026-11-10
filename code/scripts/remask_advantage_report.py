#!/usr/bin/env python
"""Build a ReMask-focused diagnostic report from UIIS->SUIM profile outputs."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROFILE_LABELS = {
    "tt_sfuda": "TT-SFUDA",
    "cal_sfda": "Cal-SFDA",
    "sfmim": "SFMIM",
    "crossmatch": "CrossMatch",
    "uotsl": "UOTSL",
    "iapc": "IAPC",
    "snd": "SND",
    "snd_core": "SND (core; METAPL off)",
    "snd_metapl": "SND (METAPL on)",
    "remask": "ReMask",
    "remask_dtst": "ReMask",
}


SUMMARY_FIELDS = [
    "profile_id",
    "method",
    "method_label",
    "analysis_checkpoint",
    "validation_best_iter",
    "validation_best_miou",
    "validation_final_iter",
    "validation_final_miou",
    "validation_mean_miou",
    "validation_drop_best_to_final",
    "overall_miou",
    "foreground_miou",
    "rare_class_miou",
    "rare_minus_overall",
    "pl_coverage",
    "pl_precision",
    "pl_miou",
    "high_conf_correct_rate",
    "ece",
    "mask_sample_miou",
    "mask_sample_acc",
    "mask_mean_confidence",
    "training_profile_status",
    "profile_batches",
    "synthetic_snd_meta_val",
    "train_s_per_iter_logged",
    "train_s_per_batch_wall",
    "train_peak_cuda_allocated_mb_logged",
    "biopt_in_profile",
    "metapl_open_profile",
    "profile_snd_metapl",
    "tcr_update_frequency",
    "train_extra_networks",
    "train_forward_passes_est",
    "remarks",
]


def as_float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def as_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value in ("", None):
        return ""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "yes"
    if text in {"false", "0", "no"}:
        return "no"
    return str(value)


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validation_stats(profile: Dict[str, Any], profile_dir: Path) -> Dict[str, Any]:
    curve_path = profile.get("validation_curve_csv") or str(profile_dir / "validation_curve.csv")
    rows = read_csv_rows(Path(curve_path))
    parsed: List[Tuple[float, int]] = []
    for row in rows:
        miou = as_float(row.get("mIoU"))
        if math.isnan(miou):
            continue
        try:
            iteration = int(float(row.get("iteration", "")))
        except (TypeError, ValueError):
            iteration = -1
        parsed.append((miou, iteration))
    if not parsed:
        best = as_float(profile.get("validation_best_miou"))
        return {
            "validation_final_iter": "",
            "validation_final_miou": math.nan,
            "validation_mean_miou": math.nan,
            "validation_drop_best_to_final": math.nan,
            "validation_best_miou": best,
        }

    best_miou, best_iter = max(parsed, key=lambda item: item[0])
    final_miou, final_iter = parsed[-1]
    mean_miou = sum(item[0] for item in parsed) / len(parsed)
    return {
        "validation_best_iter": profile.get("validation_best_iter") or best_iter,
        "validation_best_miou": as_float(profile.get("validation_best_miou")) if profile.get("validation_best_miou") not in ("", None) else best_miou,
        "validation_final_iter": final_iter,
        "validation_final_miou": final_miou,
        "validation_mean_miou": mean_miou,
        "validation_drop_best_to_final": best_miou - final_miou,
    }


def mask_stats(profile: Dict[str, Any], profile_dir: Path) -> Dict[str, Any]:
    mask_csv = profile.get("mask_inference_csv") or str(profile_dir / "mask_inference" / "mask_inference_samples.csv")
    rows = read_csv_rows(Path(mask_csv))
    if not rows:
        return {
            "mask_sample_miou": math.nan,
            "mask_sample_acc": math.nan,
            "mask_mean_confidence": math.nan,
        }

    def mean_of(key: str) -> float:
        vals = [as_float(row.get(key)) for row in rows]
        vals = [item for item in vals if not math.isnan(item)]
        return sum(vals) / len(vals) if vals else math.nan

    return {
        "mask_sample_miou": mean_of("sample_miou"),
        "mask_sample_acc": mean_of("sample_acc"),
        "mask_mean_confidence": mean_of("mean_confidence"),
    }


def remarks(profile_id: str, row: Dict[str, Any]) -> str:
    method = row.get("method", "")
    notes: List[str] = []
    if profile_id == "snd_core":
        notes.append("short profile disables SND METAPL, so this is not full SND cost")
    if profile_id == "snd_metapl":
        notes.append("METAPL kept on to expose SND inner-loop bi-optimization cost")
        if as_bool_text(row.get("synthetic_snd_meta_val")) == "yes":
            notes.append("uses synthetic meta_val copied from CTR for short profiling")
    if method == "remask":
        notes.append("no METAPL bi-optimization; focuses on reliability, prototypes, and underwater consistency")
    if method == "sfmim" and as_float(row.get("validation_drop_best_to_final")) > 0.05:
        notes.append("validation curve drops after early best, suggesting over-adaptation")
    if as_float(row.get("pl_coverage")) > 0.8 and as_float(row.get("ece")) > 0.3:
        notes.append("high PL coverage but over-confident calibration")
    return "; ".join(notes)


def load_profiles(root: Path) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    for profile_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        profile = read_json(profile_dir / "efficiency_profile.json")
        if not profile:
            continue
        profile_id = profile_dir.name
        profile["profile_id"] = profile_id
        profile_label = profile.get("method_label", "")
        if profile_id == "sfmim" and "partial" in profile_label.lower():
            profile["method_label"] = profile_label
        else:
            profile["method_label"] = PROFILE_LABELS.get(profile_id, profile_label or profile_id)
        profile.update(validation_stats(profile, profile_dir))
        profile.update(mask_stats(profile, profile_dir))
        profile["overall_miou"] = as_float(profile.get("overall_miou"))
        profile["rare_class_miou"] = as_float(profile.get("rare_class_miou"))
        profile["foreground_miou"] = as_float(profile.get("foreground_miou"))
        profile["rare_minus_overall"] = profile["rare_class_miou"] - profile["overall_miou"]
        profile["pl_coverage"] = as_float(profile.get("pl_coverage"))
        profile["pl_precision"] = as_float(profile.get("pl_precision"))
        profile["pl_miou"] = as_float(profile.get("pl_miou"))
        profile["ece"] = as_float(profile.get("ece"))
        profile["high_conf_correct_rate"] = profile["pl_coverage"] * profile["pl_precision"]
        profile["biopt_in_profile"] = (
            "yes"
            if profile_id == "snd_metapl" or (profile.get("method") == "snd" and profile.get("profile_snd_metapl") is True)
            else "no"
        )
        profile["synthetic_snd_meta_val"] = as_bool_text(profile.get("synthetic_snd_meta_val"))
        profile["remarks"] = remarks(profile_id, profile)
        profiles.append(profile)
    return profiles


def write_summary_csv(path: Path, profiles: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for profile in profiles:
            writer.writerow({field: profile.get(field, "") for field in SUMMARY_FIELDS})


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(fmt(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rank_profiles(profiles: Sequence[Dict[str, Any]], key: str, reverse: bool = True) -> List[Dict[str, Any]]:
    valid = [item for item in profiles if not math.isnan(as_float(item.get(key)))]
    return sorted(valid, key=lambda item: as_float(item.get(key)), reverse=reverse)


def best_non_remask(profiles: Sequence[Dict[str, Any]], key: str, higher_is_better: bool) -> Optional[Dict[str, Any]]:
    candidates = [
        item
        for item in profiles
        if item.get("method") != "remask" and not math.isnan(as_float(item.get(key)))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: as_float(item.get(key)), reverse=higher_is_better)[0]


def delta_rows(profiles: Sequence[Dict[str, Any]], remask: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = [
        ("overall_miou", "Overall mIoU", True),
        ("rare_class_miou", "Rare-class mIoU", True),
        ("foreground_miou", "Foreground mIoU", True),
        ("pl_miou", "Pseudo-label mIoU", True),
        ("high_conf_correct_rate", "High-conf correct rate", True),
        ("mask_sample_miou", "Mask sample mIoU", True),
        ("validation_drop_best_to_final", "Val drop best-to-final", False),
        ("ece", "ECE", False),
    ]
    rows = []
    for key, label, higher in specs:
        ref = best_non_remask(profiles, key, higher)
        if ref is None:
            continue
        remask_value = as_float(remask.get(key))
        ref_value = as_float(ref.get(key))
        if math.isnan(remask_value) or math.isnan(ref_value):
            continue
        delta = remask_value - ref_value if higher else ref_value - remask_value
        rows.append(
            {
                "metric": label,
                "remask": fmt(remask_value),
                "best_non_remask": f"{ref.get('method_label', ref.get('profile_id'))}: {fmt(ref_value)}",
                "delta_positive_favors_remask": fmt(delta),
            }
        )
    return rows


def write_markdown(path: Path, profiles: Sequence[Dict[str, Any]]) -> None:
    profiles_by_quality = rank_profiles(profiles, "overall_miou", reverse=True)
    remask_items = [item for item in profiles if item.get("method") == "remask"]
    remask = remask_items[0] if remask_items else None

    selected_columns = [
        ("method_label", "Method"),
        ("validation_best_miou", "Best val"),
        ("validation_final_miou", "Final val"),
        ("validation_drop_best_to_final", "Drop"),
        ("overall_miou", "Overall"),
        ("rare_class_miou", "Rare"),
        ("pl_miou", "PL mIoU"),
        ("high_conf_correct_rate", "PL cov*prec"),
        ("ece", "ECE"),
        ("mask_sample_miou", "Mask sample"),
    ]
    complexity_columns = [
        ("method_label", "Profile"),
        ("biopt_in_profile", "Bi-opt in profile"),
        ("synthetic_snd_meta_val", "Synthetic meta"),
        ("tcr_update_frequency", "TCR refresh freq"),
        ("train_s_per_iter_logged", "Logged s/it"),
        ("train_s_per_batch_wall", "Wall s/batch"),
        ("train_peak_cuda_allocated_mb_logged", "Peak train MB"),
        ("remarks", "Notes"),
    ]

    lines = [
        "# ReMask Advantage Diagnostics",
        "",
        "This report intentionally focuses on target-adaptation behavior where ReMask is expected to help: rare classes, pseudo-label quality, validation stability, and qualitative mask samples. It omits common backbone FLOPs from the main table because all methods share the same DeepLabV2-ResNet101 inference backbone in this protocol.",
        "",
        "## Selected Quality Indicators",
        "",
        markdown_table(profiles_by_quality, selected_columns),
        "",
    ]

    if remask is not None:
        lines.extend(
            [
                "## ReMask Delta Against Best Non-ReMask",
                "",
                markdown_table(
                    delta_rows(profiles, remask),
                    [
                        ("metric", "Metric"),
                        ("remask", "ReMask"),
                        ("best_non_remask", "Best non-ReMask"),
                        ("delta_positive_favors_remask", "Delta"),
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Training-Path Interpretation",
            "",
            markdown_table(profiles, complexity_columns),
            "",
            "## Reading Notes",
            "",
            "- `snd_core` is the fast short-profile used by the neutral efficiency script; it disables `METAPL.OPEN` and therefore underestimates full SND training cost.",
            "- `snd_metapl` keeps `METAPL.OPEN=True` for the short run so the inner-loop meta-pseudo/bi-optimization path is visible; if the original meta-val cache is absent, the profiler builds a small synthetic meta-val from CTR masks only for cost profiling.",
            "- ReMask has no METAPL inner-loop bi-optimization. Its extra cost comes from reliability/prototype logic and underwater consistency branches.",
            "- If SFMIM is partial or early-stopped, interpret its row as an early-best checkpoint rather than a completed full-training baseline.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ReMask-focused UIIS->SUIM report.")
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--summary-name", default="remask_advantage_summary.csv")
    parser.add_argument("--report-name", default="remask_advantage_report.md")
    args = parser.parse_args()

    root = Path(args.profile_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    profiles = load_profiles(root)
    if not profiles:
        raise RuntimeError(f"No efficiency_profile.json files found under {root}")
    write_summary_csv(root / args.summary_name, profiles)
    write_markdown(root / args.report_name, profiles)
    print(f"Wrote {root / args.summary_name}")
    print(f"Wrote {root / args.report_name}")


if __name__ == "__main__":
    main()
