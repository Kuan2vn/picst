#!/usr/bin/env python3
"""Generate detailed LaTeX result tables from underwater evidence manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CLASS_COLUMNS = [
    (0, "Bg."),
    (1, "Diver"),
    (2, "Plant"),
    (3, "Wreck"),
    (4, "Robot"),
    (5, "Reef"),
    (6, "Fish"),
    (7, "Floor"),
]

SETTING_LABELS = {
    "uiis_train->suim_val": "UIIS $\\rightarrow$ SUIM",
    "usis10k_bg_train->suim_val": "USIS10K $\\rightarrow$ SUIM",
}

METHOD_META = {
    "tt_sfuda": {
        "label": "TT-SFUDA (MM'21)",
        "year": 2021,
        "order": 10,
    },
    "cal_sfda": {
        "label": "Cal-SFDA (MM'23)",
        "year": 2023,
        "order": 20,
    },
    "sfmim": {
        "label": "SFMIM (MM'23)",
        "year": 2023,
        "order": 21,
    },
    "crossmatch": {
        "label": "CrossMatch (ICCV'23)",
        "year": 2023,
        "order": 22,
    },
    "uotsl": {
        "label": "UOTSL (TIP'23)",
        "year": 2023,
        "order": 23,
    },
    "crots": {
        "label": "CROTS (IJCV'24)",
        "year": 2024,
        "order": 30,
    },
    "iapc": {
        "label": "IAPC (TIV'24)",
        "year": 2024,
        "order": 31,
    },
    "snd": {
        "label": "SND (CVPR'24)",
        "year": 2024,
        "order": 40,
    },
    "dtst_warm": {
        "label": "DTST-warm (CVPR'23)",
        "year": 2023,
        "order": 15,
    },
    "dtst_dynamic": {
        "label": "DTST-dynamic (CVPR'23)",
        "year": 2023,
        "order": 16,
    },
    "super_dtst": {
        "label": "Super-DTST (CVPR'23)",
        "year": 2023,
        "order": 17,
    },
    "snd_dtst": {
        "label": "DTST + SND (CVPR'24)",
        "year": 2024,
        "order": 44,
    },
    "remask": {
        "label": "ReMask (Ours)",
        "year": 2026,
        "order": 90,
    },
    "remask_dtst": {
        "label": "ReMask + DTST (Ours)",
        "year": 2026,
        "order": 91,
    },
    "remask_superdtst": {
        "label": "ReMask + Super-DTST (Ours)",
        "year": 2026,
        "order": 92,
    },
}

EXPECTED_METHODS_BY_SETTING = {
    "uiis_train->suim_val": [
        "tt_sfuda",
        "cal_sfda",
        "sfmim",
        "crossmatch",
        "uotsl",
        "crots",
        "iapc",
        "snd",
        "dtst_warm",
        "super_dtst",
        "snd_dtst",
        "remask",
        "remask_dtst",
        "remask_superdtst",
    ],
    "usis10k_bg_train->suim_val": [
        "tt_sfuda",
        "cal_sfda",
        "sfmim",
        "crossmatch",
        "uotsl",
        "crots",
        "iapc",
        "snd",
        "dtst_warm",
        "dtst_dynamic",
        "snd_dtst",
        "remask",
        "remask_dtst",
    ],
}


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _value(row: Optional[Dict[str, Any]], metric: str, class_id: Optional[int] = None) -> Optional[float]:
    if row is None:
        return None
    if class_id is None:
        key = "best_mIoU" if metric == "iou" else "best_mAcc"
        value = row.get(key)
        return float(value) if value is not None and value != "" else None
    for item in row.get("classes", []):
        if int(item.get("id", -1)) == class_id:
            key = "iou" if metric == "iou" else "accuracy"
            value = item.get(key)
            return float(value) if value is not None and value != "" else None
    return None


def _unique_ranks(rows: Sequence[Optional[Dict[str, Any]]], metric: str, class_id: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    values = []
    for row in rows:
        value = _value(row, metric, class_id)
        if value is not None and not math.isnan(value):
            values.append(round(value, 10))
    unique = sorted(set(values), reverse=True)
    best = unique[0] if unique else None
    second = unique[1] if len(unique) > 1 else None
    return best, second


def _format_value(
    value: Optional[float],
    best: Optional[float],
    second: Optional[float],
) -> str:
    if value is None or math.isnan(value):
        return "--"
    rounded = round(value, 10)
    text = "{:.4f}".format(value)
    if best is not None and rounded == best:
        return "\\textbf{" + text + "}"
    if second is not None and rounded == second:
        return "\\underline{" + text + "}"
    return text


def _method_sort_key(method: str) -> Tuple[int, int, str]:
    meta = METHOD_META.get(method, {})
    return int(meta.get("year", 9999)), int(meta.get("order", 9999)), method


def _method_label(method: str) -> str:
    return METHOD_META.get(method, {}).get("label", method.replace("_", "\\_"))


def _build_setting_rows(manifest: Sequence[Dict[str, Any]], setting: str) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    by_method = {row["method"]: row for row in manifest if row.get("setting") == setting}
    methods = list(EXPECTED_METHODS_BY_SETTING.get(setting, []))
    for method in by_method:
        if method not in methods:
            methods.append(method)
    methods.sort(key=_method_sort_key)
    return [(method, by_method.get(method)) for method in methods]


def _render_table(
    manifest: Sequence[Dict[str, Any]],
    metric: str,
    caption: str,
    label: str,
) -> str:
    metric_header = "mIoU" if metric == "iou" else "mAcc"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llc" + "c" * len(CLASS_COLUMNS) + "c}",
        "\\toprule",
        "Datasets & Methods & Iter & "
        + " & ".join(name for _, name in CLASS_COLUMNS)
        + " & "
        + metric_header
        + " \\\\",
        "\\midrule",
    ]

    for setting_index, setting in enumerate(SETTING_LABELS):
        setting_rows = _build_setting_rows(manifest, setting)
        numeric_rows = [row for _, row in setting_rows]
        ranks: Dict[Optional[int], Tuple[Optional[float], Optional[float]]] = {}
        for class_id, _ in CLASS_COLUMNS:
            ranks[class_id] = _unique_ranks(numeric_rows, metric, class_id)
        ranks[None] = _unique_ranks(numeric_rows, metric, None)

        if setting_index > 0:
            lines.append("\\midrule")
        multirow = "\\multirow{" + str(len(setting_rows)) + "}{*}{" + SETTING_LABELS[setting] + "}"
        for row_index, (method, row) in enumerate(setting_rows):
            dataset_cell = multirow if row_index == 0 else ""
            iter_cell = str(row.get("best_iteration", "--")) if row is not None else "--"
            values = []
            for class_id, _ in CLASS_COLUMNS:
                best, second = ranks[class_id]
                values.append(_format_value(_value(row, metric, class_id), best, second))
            best, second = ranks[None]
            values.append(_format_value(_value(row, metric, None), best, second))
            lines.append(
                "{} & {} & {} & {} \\\\".format(
                    dataset_cell,
                    _method_label(method),
                    iter_cell,
                    " & ".join(values),
                )
            )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def _render_method_notes() -> str:
    rows = sorted(METHOD_META.items(), key=lambda item: _method_sort_key(item[0]))
    lines = [
        "% Method chronology used in the result tables:",
    ]
    for method, meta in rows:
        lines.append("% - {}: {}".format(method, meta["label"]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detailed LaTeX underwater SFUDA result tables.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results_underwater") / "evidence_summary" / "best_iters_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_underwater") / "evidence_summary" / "latex_detailed_results_tables.tex",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    text = "\n\n".join(
        [
            _render_method_notes(),
            _render_table(
                manifest,
                "iou",
                "Class-wise IoU and mean IoU on underwater SFUDA settings. Best and second-best values within each dataset are bolded and underlined, respectively.",
                "tab:underwater_class_iou",
            ),
            _render_table(
                manifest,
                "acc",
                "Class-wise accuracy and mean accuracy on underwater SFUDA settings. Best and second-best values within each dataset are bolded and underlined, respectively.",
                "tab:underwater_class_acc",
            ),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
