#!/usr/bin/env python
"""Create paper-ready ReMask main-result tables from the latest profile run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


MAIN_METHODS = [
    "tt_sfuda",
    "cal_sfda",
    "sfmim",
    "crossmatch",
    "uotsl",
    "iapc",
    "snd_metapl",
    "remask",
]

METHOD_ORDER = {
    "tt_sfuda": "TT-SFUDA",
    "cal_sfda": "Cal-SFDA",
    "sfmim": "SFMIM",
    "crossmatch": "CrossMatch",
    "uotsl": "UOTSL",
    "iapc": "IAPC",
    "snd_metapl": "SND",
    "remask": "ReMask",
}

TAIL_CLASSES = [
    "Human divers",
    "Aquatic plants/sea-grass",
    "Robots/instruments",
]

ALL_CLASSES = [
    "Background",
    "Human divers",
    "Aquatic plants/sea-grass",
    "Wrecks/ruins",
    "Robots/instruments",
    "Reefs and invertebrates",
    "Fish and vertebrates",
    "Sea-floor and rocks",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Any) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "--"
    return f"{number:.{digits}f}"


def pct_gain(delta: float) -> str:
    if math.isnan(delta):
        return "--"
    return f"{delta:+.4f}"


def escape_latex(text: Any) -> str:
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def value_with_rank(row: Dict[str, Any], key: str, rows: Sequence[Dict[str, Any]], higher: bool = True) -> str:
    values = sorted(
        {as_float(item.get(key)) for item in rows if not math.isnan(as_float(item.get(key)))},
        reverse=higher,
    )
    value = as_float(row.get(key))
    text = fmt(value)
    if math.isnan(value) or not values:
        return text
    if abs(value - values[0]) < 1e-12:
        return f"**{text}**"
    if len(values) > 1 and abs(value - values[1]) < 1e-12:
        return f"<u>{text}</u>"
    return text


def latex_value_with_rank(row: Dict[str, Any], key: str, rows: Sequence[Dict[str, Any]], higher: bool = True) -> str:
    values = sorted(
        {as_float(item.get(key)) for item in rows if not math.isnan(as_float(item.get(key)))},
        reverse=higher,
    )
    value = as_float(row.get(key))
    text = fmt(value)
    if math.isnan(value) or not values:
        return text
    if abs(value - values[0]) < 1e-12:
        return rf"\textbf{{{text}}}"
    if len(values) > 1 and abs(value - values[1]) < 1e-12:
        return rf"\underline{{{text}}}"
    return text


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def latex_table(caption: str, label: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    colspec = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(escape_latex(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def load_summary(root: Path) -> List[Dict[str, Any]]:
    rows = [row for row in read_csv(root / "remask_advantage_summary.csv") if row.get("profile_id") in MAIN_METHODS]
    rows.sort(key=lambda row: MAIN_METHODS.index(row["profile_id"]))
    for row in rows:
        profile_label = row.get("method_label", "")
        if row["profile_id"] == "sfmim" and "partial" in profile_label.lower():
            row["display_method"] = profile_label
        else:
            row["display_method"] = METHOD_ORDER[row["profile_id"]]
    return rows


def load_class_metrics(root: Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for row in rows:
        payload = read_json(root / row["profile_id"] / "extra_analysis.json")
        metrics[row["profile_id"]] = {
            item["class_name"]: as_float(item.get("iou")) for item in payload.get("class_metrics", [])
        }
    return metrics


def enrich_with_extra_analysis(root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    keys = [
        "bg_iou",
        "head_miou",
        "medium_miou",
        "tail_miou",
        "mean_entropy",
        "confidence_gap",
    ]
    for row in rows:
        payload = read_json(root / row["profile_id"] / "extra_analysis.json")
        for key in keys:
            if row.get(key) in ("", None):
                row[key] = payload.get(key, "")


def best_non_remask(rows: Sequence[Dict[str, Any]], key: str, higher: bool = True) -> Tuple[str, float]:
    candidates = [row for row in rows if row["profile_id"] != "remask"]
    candidates = [row for row in candidates if not math.isnan(as_float(row.get(key)))]
    if not candidates:
        return "--", math.nan
    best = sorted(candidates, key=lambda row: as_float(row[key]), reverse=higher)[0]
    return best["display_method"], as_float(best[key])


def build_delta_table(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    remask = next(row for row in rows if row["profile_id"] == "remask")
    specs = [
        ("overall_miou", "Overall mIoU", True),
        ("foreground_miou", "Foreground mIoU", True),
        ("rare_class_miou", "Rare-class mIoU", True),
        ("pl_miou", "Pseudo-label mIoU", True),
        ("high_conf_correct_rate", "High-conf correct rate", True),
        ("mask_sample_miou", "Mask sample mIoU", True),
        ("validation_final_miou", "Final validation mIoU", True),
    ]
    headers = ["Metric", "ReMask", "Best non-ReMask", "Gain"]
    md_rows: List[List[str]] = []
    tex_rows: List[List[str]] = []
    for key, label, higher in specs:
        ref_name, ref_value = best_non_remask(rows, key, higher=higher)
        remask_value = as_float(remask[key])
        gain = (remask_value - ref_value) if higher else (ref_value - remask_value)
        md_rows.append([label, fmt(remask_value), f"{ref_name} {fmt(ref_value)}", pct_gain(gain)])
        tex_rows.append([escape_latex(label), fmt(remask_value), escape_latex(f"{ref_name} {fmt(ref_value)}"), pct_gain(gain)])
    return headers, md_rows, tex_rows


def build_compact_quality(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    headers = ["Method", "Overall", "FG", "Rare", "PL mIoU", "PL CxP", "Mask", "Final val"]
    keys = [
        ("overall_miou", True),
        ("foreground_miou", True),
        ("rare_class_miou", True),
        ("pl_miou", True),
        ("high_conf_correct_rate", True),
        ("mask_sample_miou", True),
        ("validation_final_miou", True),
    ]
    md_rows = []
    tex_rows = []
    for row in rows:
        md_rows.append([row["display_method"]] + [value_with_rank(row, key, rows, higher) for key, higher in keys])
        tex_rows.append([escape_latex(row["display_method"])] + [latex_value_with_rank(row, key, rows, higher) for key, higher in keys])
    return headers, md_rows, tex_rows


def build_rare_table(rows: Sequence[Dict[str, Any]], class_metrics: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    headers = ["Method", "Tail mIoU", "Human", "Plants", "Robots", "Head mIoU", "Medium mIoU"]
    md_rows = []
    tex_rows = []
    tail_class_keys = [
        ("Human divers", "human_iou"),
        ("Aquatic plants/sea-grass", "plants_iou"),
        ("Robots/instruments", "robots_iou"),
    ]
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        per_class = class_metrics[row["profile_id"]]
        item["human_iou"] = per_class.get("Human divers", math.nan)
        item["plants_iou"] = per_class.get("Aquatic plants/sea-grass", math.nan)
        item["robots_iou"] = per_class.get("Robots/instruments", math.nan)
        enriched.append(item)
    keys = [
        ("rare_class_miou", True),
        ("human_iou", True),
        ("plants_iou", True),
        ("robots_iou", True),
        ("head_miou", True),
        ("medium_miou", True),
    ]
    for row in enriched:
        md_rows.append([row["display_method"]] + [value_with_rank(row, key, enriched, higher) for key, higher in keys])
        tex_rows.append([escape_latex(row["display_method"])] + [latex_value_with_rank(row, key, enriched, higher) for key, higher in keys])
    return headers, md_rows, tex_rows


def build_pseudo_table(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    headers = ["Method", "Coverage", "Precision", "PL mIoU", "Cov x Prec", "ECE down", "Entropy down"]
    keys = [
        ("pl_coverage", True),
        ("pl_precision", True),
        ("pl_miou", True),
        ("high_conf_correct_rate", True),
        ("ece", False),
        ("mean_entropy", False),
    ]
    md_headers = ["Method", "Coverage", "Precision", "PL mIoU", "Cov*Prec", "ECE down", "Entropy down"]
    md_rows = []
    tex_rows = []
    for row in rows:
        md_rows.append([row["display_method"]] + [value_with_rank(row, key, rows, higher) for key, higher in keys])
        tex_rows.append([escape_latex(row["display_method"])] + [latex_value_with_rank(row, key, rows, higher) for key, higher in keys])
    return md_headers, md_rows, [headers, tex_rows]  # type: ignore[return-value]


def build_validation_table(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    headers = ["Method", "Best iter", "Best mIoU", "Final mIoU", "Mean mIoU", "Drop down"]
    keys = [
        ("validation_best_miou", True),
        ("validation_final_miou", True),
        ("validation_mean_miou", True),
        ("validation_drop_best_to_final", False),
    ]
    md_rows = []
    tex_rows = []
    for row in rows:
        md_rows.append(
            [row["display_method"], str(int(as_float(row["validation_best_iter"])))]
            + [value_with_rank(row, key, rows, higher) for key, higher in keys]
        )
        tex_rows.append(
            [escape_latex(row["display_method"]), str(int(as_float(row["validation_best_iter"])))]
            + [latex_value_with_rank(row, key, rows, higher) for key, higher in keys]
        )
    return headers, md_rows, tex_rows


def build_training_table(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    selected = [row for row in rows if row["profile_id"] in {"snd_metapl", "remask", "sfmim", "cal_sfda"}]
    selected.sort(key=lambda row: ["cal_sfda", "sfmim", "snd_metapl", "remask"].index(row["profile_id"]))
    headers = ["Method", "Bi-opt", "Logged s/it", "Wall s/batch", "Peak MB", "Quality mIoU"]
    keys = [
        ("train_s_per_iter_logged", False),
        ("train_s_per_batch_wall", False),
        ("train_peak_cuda_allocated_mb_logged", False),
        ("overall_miou", True),
    ]
    md_rows = []
    tex_rows = []
    for row in selected:
        md_rows.append(
            [row["display_method"], row.get("biopt_in_profile", "")]
            + [value_with_rank(row, key, selected, higher) for key, higher in keys]
        )
        tex_rows.append(
            [escape_latex(row["display_method"]), escape_latex(row.get("biopt_in_profile", ""))]
            + [latex_value_with_rank(row, key, selected, higher) for key, higher in keys]
        )
    return headers, md_rows, tex_rows


def build_full_class_table(rows: Sequence[Dict[str, Any]], class_metrics: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    headers = ["Method"] + [name.replace("Aquatic plants/sea-grass", "Plants").replace("Robots/instruments", "Robots").replace("Fish and vertebrates", "Fish").replace("Sea-floor and rocks", "Sea-floor").replace("Reefs and invertebrates", "Reefs").replace("Wrecks/ruins", "Wrecks") for name in ALL_CLASSES]
    enriched = []
    for row in rows:
        item = dict(row)
        per_class = class_metrics[row["profile_id"]]
        for class_name in ALL_CLASSES:
            item[class_name] = per_class.get(class_name, math.nan)
        enriched.append(item)
    md_rows = []
    tex_rows = []
    for row in enriched:
        md_rows.append([row["display_method"]] + [value_with_rank(row, class_name, enriched, True) for class_name in ALL_CLASSES])
        tex_rows.append([escape_latex(row["display_method"])] + [latex_value_with_rank(row, class_name, enriched, True) for class_name in ALL_CLASSES])
    return headers, md_rows, tex_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", required=True)
    args = parser.parse_args()

    root = Path(args.profile_root)
    rows = load_summary(root)
    enrich_with_extra_analysis(root, rows)
    class_metrics = load_class_metrics(root, rows)

    table_defs = []

    def add_table(title: str, label: str, caption: str, payload: Tuple[List[str], List[List[str]], List[List[str]]]) -> None:
        headers, md_rows, tex_rows = payload
        table_defs.append((title, label, caption, headers, md_rows, headers, tex_rows))

    add_table("Main compact result", "tab:remask_compact", "Compact main comparison on UIIS to SUIM.", build_compact_quality(rows))
    add_table("ReMask gain over best competitor", "tab:remask_gain", "ReMask gain against the best non-ReMask method for each diagnostic.", build_delta_table(rows))
    add_table("Rare-class robustness", "tab:remask_rare", "Tail-class and class-frequency robustness on SUIM.", build_rare_table(rows, class_metrics))
    pseudo_headers, pseudo_md_rows, pseudo_tex_bundle = build_pseudo_table(rows)
    pseudo_tex_headers, pseudo_tex_rows = pseudo_tex_bundle
    table_defs.append(("Pseudo-label quality", "tab:remask_pl", "Pseudo-label reliability at confidence threshold 0.9.", pseudo_headers, pseudo_md_rows, pseudo_tex_headers, pseudo_tex_rows))
    add_table("Validation stability", "tab:remask_stability", "Validation-curve stability across target adaptation.", build_validation_table(rows))
    add_table("Training-path diagnostic", "tab:remask_trainpath", "Training-path diagnostic showing SND bi-optimization and ReMask quality/cost trade-off.", build_training_table(rows))
    add_table("Class-wise IoU", "tab:remask_classwise", "Class-wise IoU on SUIM validation.", build_full_class_table(rows, class_metrics))

    md_lines = [
        "# Main-Result Tables for ReMask",
        "",
        "Source run: `{}`".format(root.as_posix()),
        "",
        "Use the first three tables in the main results if space is tight. The stability, training-path, and full class-wise tables are good candidates for ablation/analysis or supplementary material.",
        "",
        "Important consistency note: these tables use the checkpoint/protocol from `remask_advantage_summary.csv`. If the camera-ready main table uses a different ReMask checkpoint, regenerate this file with the same checkpoint before reporting final numbers.",
        "",
    ]
    tex_blocks: List[str] = []
    for idx, (title, label, caption, md_headers, md_rows, tex_headers, tex_rows) in enumerate(table_defs, start=1):
        md_lines.extend([f"## Table {idx}. {title}", "", md_table(md_headers, md_rows), ""])
        tex_blocks.append(latex_table(caption, label, tex_headers, tex_rows))

    md_path = root / "main_results_remask_tables.md"
    tex_path = root / "main_results_remask_tables.tex"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    tex_path.write_text("\n\n".join(tex_blocks) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
