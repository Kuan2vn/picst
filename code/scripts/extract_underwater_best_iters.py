#!/usr/bin/env python3
"""Extract best validation iterations and report-ready evidence blocks.

This script is intentionally log-first.  It can run on a copied results folder
after large checkpoint files have been removed, then produce a manifest that a
separate evaluator can use on HPC when the checkpoints are present again.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SFDA_METHODS = {
    "source_only",
    "cal_sfda",
    "tt_sfuda",
    "crossmatch",
    "crots",
    "iapc",
    "sfmim",
    "uotsl",
}

METHOD_ORDER = [
    "source_only",
    "dtst_warm",
    "dtst_dynamic",
    "super_dtst",
    "snd",
    "snd_dtst",
    "remask",
    "remask_dtst",
    "remask_superdtst",
    "cal_sfda",
    "tt_sfuda",
    "crossmatch",
    "crots",
    "iapc",
    "sfmim",
    "uotsl",
]

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VAL_RE = re.compile(
    r"Val result(?: iter (?P<iter>\d+))?: mIoU/mAcc/allAcc "
    r"(?P<miou>{0})/(?P<macc>{0})/(?P<allacc>{0})".format(FLOAT_PATTERN)
)
CLASS_RE = re.compile(
    r"Class_(?P<id>\d+)\s+(?P<name>.*?)\s+Result: iou/accuracy "
    r"(?P<iou>{0})/(?P<acc>{0})".format(FLOAT_PATTERN)
)
TRAIN_ITER_RE = re.compile(r"\biter:\s*(?P<iter>\d+)\b")


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _last_regex(pattern: str, text: str, flags: int = re.MULTILINE) -> Optional[str]:
    matches = re.findall(pattern, text, flags)
    if not matches:
        return None
    item = matches[-1]
    if isinstance(item, tuple):
        return next((part for part in item if part), None)
    return item


def _parse_namespace_line(text: str) -> Dict[str, Any]:
    line = next((ln for ln in text.splitlines() if "Namespace(" in ln), "")
    meta: Dict[str, Any] = {}
    if not line:
        return meta

    def pick_string(name: str) -> Optional[str]:
        match = re.search(r"\b{}\='([^']*)'".format(re.escape(name)), line)
        return match.group(1) if match else None

    def pick_bool(name: str) -> Optional[bool]:
        match = re.search(r"\b{}=(True|False)".format(re.escape(name)), line)
        if not match:
            return None
        return match.group(1) == "True"

    for key in ("config_file", "method"):
        value = pick_string(key)
        if value is not None:
            meta[key] = value
    for key in ("seafloor_ignore", "suim_legacy_color_map", "paper_preset"):
        value = pick_bool(key)
        if value is not None:
            meta[key] = value

    opts_match = re.search(r"\bopts=(\[.*?\])(?:,\s*distributed=|\))", line)
    if opts_match:
        try:
            meta["opts"] = ast.literal_eval(opts_match.group(1))
        except (SyntaxError, ValueError):
            meta["opts"] = []
    return meta


def _get_opt(opts: Sequence[Any], key: str) -> Optional[str]:
    for index, token in enumerate(opts[:-1]):
        if str(token) == key:
            return str(opts[index + 1])
    return None


def _method_from_dir(run_dir: Path, namespace_method: Optional[str]) -> str:
    if namespace_method:
        return namespace_method
    name = run_dir.name.lower()
    for method in sorted(METHOD_ORDER, key=len, reverse=True):
        if name == method or name.startswith(method + "_") or ("_" + method + "_") in name:
            return method
    if "remask_superdtst" in name:
        return "remask_superdtst"
    if "remask_dtst" in name:
        return "remask_dtst"
    if "remask" in name:
        return "remask"
    if "snd_dtst" in name:
        return "snd_dtst"
    if "snd" in name:
        return "snd"
    if "dtst_dynamic" in name:
        return "dtst_dynamic"
    if "dtst_warm" in name:
        return "dtst_warm"
    if "super_dtst" in name:
        return "super_dtst"
    return run_dir.name


def _family_for_method(method: str, run_dir: Path, text: str) -> str:
    if method in SFDA_METHODS or "SFDA.baselines" in text or (run_dir / "metrics.csv").exists():
        return "sfda_baseline"
    if method.startswith("remask"):
        return "remask"
    return "snd"


def _setting_name(source_train: str, target_train: str, test_set: str) -> str:
    source = source_train or "unknown_source"
    target = test_set or target_train or "unknown_target"
    return "{}->{}".format(source, target)


def _short_run_id(results_root: Path, run_dir: Path) -> str:
    try:
        rel = run_dir.relative_to(results_root)
    except ValueError:
        rel = run_dir
    return str(rel).replace("\\", "/")


def _parse_run_metadata(results_root: Path, run_dir: Path, text: str) -> Dict[str, Any]:
    ns = _parse_namespace_line(text)
    opts = [str(item) for item in ns.get("opts", [])]

    output_dir = (
        _get_opt(opts, "OUTPUT_DIR")
        or _last_regex(r"^OUTPUT_DIR:\s*(.*?)\s*$", text)
        or ""
    )
    source_train = _get_opt(opts, "DATASETS.SOURCE_TRAIN") or ""
    target_train = _get_opt(opts, "DATASETS.TARGET_TRAIN") or ""
    test_set = _get_opt(opts, "DATASETS.TEST") or ""

    method = _method_from_dir(run_dir, ns.get("method"))
    family = _family_for_method(method, run_dir, text)
    checkpoint_period = _int(
        _get_opt(opts, "SOLVER.CHECKPOINT_PERIOD")
        or _last_regex(r"^\s*CHECKPOINT_PERIOD:\s*(\d+)\s*$", text)
    )
    max_iter = _int(
        _get_opt(opts, "SOLVER.MAX_ITER")
        or _last_regex(r"^\s*MAX_ITER:\s*(\d+)\s*$", text)
    )
    stop_iter = _int(
        _get_opt(opts, "SOLVER.STOP_ITER")
        or _last_regex(r"^\s*STOP_ITER:\s*(\d+)\s*$", text)
    )

    return {
        "run_id": _short_run_id(results_root, run_dir),
        "method": method,
        "family": family,
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "log.txt"),
        "config_file": ns.get("config_file", ""),
        "opts": opts,
        "output_dir_logged": output_dir,
        "source_train": source_train,
        "target_train": target_train,
        "test_set": test_set,
        "setting": _setting_name(source_train, target_train, test_set),
        "checkpoint_period": checkpoint_period,
        "max_iter": max_iter,
        "stop_iter": stop_iter,
        "seafloor_ignore": bool(ns.get("seafloor_ignore", False)),
        "suim_legacy_color_map": bool(ns.get("suim_legacy_color_map", False)),
        "paper_preset": ns.get("paper_preset"),
    }


def _classes_from_csv_row(row: Dict[str, str]) -> List[Dict[str, Any]]:
    classes: List[Dict[str, Any]] = []
    index = 0
    while "class_{}_iou".format(index) in row:
        classes.append(
            {
                "id": index,
                "name": row.get("class_{}_name".format(index), str(index)),
                "iou": _float(row.get("class_{}_iou".format(index))),
                "accuracy": _float(
                    row.get("class_{}_acc".format(index), row.get("class_{}_accuracy".format(index)))
                ),
            }
        )
        index += 1
    return classes


def _read_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "iteration": _int(row.get("iteration"), 0) or 0,
                    "method": row.get("method", ""),
                    "checkpoint": row.get("checkpoint", ""),
                    "mIoU": _float(row.get("mIoU")),
                    "mAcc": _float(row.get("mAcc")),
                    "allAcc": _float(row.get("allAcc")),
                    "classes": _classes_from_csv_row(row),
                    "metric_source": str(path),
                    "iteration_source": "metrics.csv",
                }
            )
    return rows


def _read_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                {
                    "iteration": _int(item.get("iteration"), 0) or 0,
                    "method": item.get("method", ""),
                    "checkpoint": item.get("checkpoint", ""),
                    "mIoU": _float(item.get("mIoU")),
                    "mAcc": _float(item.get("mAcc")),
                    "allAcc": _float(item.get("allAcc")),
                    "classes": item.get("classes", []),
                    "metric_source": str(path),
                    "iteration_source": "metrics.jsonl",
                }
            )
    return rows


def _infer_eval_iter(
    last_train_iter: Optional[int],
    checkpoint_period: Optional[int],
    max_iter: Optional[int],
    stop_iter: Optional[int],
    val_index: int,
) -> Tuple[int, str]:
    limits = [value for value in (max_iter, stop_iter) if value is not None]
    train_limit = min(limits) if limits else None
    if last_train_iter is not None:
        if train_limit is not None and last_train_iter >= train_limit:
            return train_limit, "inferred_final_from_trainer_iter"
        if checkpoint_period and checkpoint_period > 0:
            inferred = ((last_train_iter // checkpoint_period) + 1) * checkpoint_period - 1
            if train_limit is not None:
                inferred = min(inferred, train_limit)
            return max(inferred, last_train_iter), "inferred_checkpoint_period_minus_one"
        return last_train_iter, "inferred_previous_trainer_iter"
    if checkpoint_period and checkpoint_period > 0:
        return val_index * checkpoint_period - 1, "sequence_checkpoint_period_minus_one"
    return val_index, "sequence_index"


def _parse_log_metrics(text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    last_train_iter: Optional[int] = None
    current: Optional[Dict[str, Any]] = None

    def flush_current() -> None:
        nonlocal current
        if current is not None:
            rows.append(current)
            current = None

    for line in text.splitlines():
        train_match = TRAIN_ITER_RE.search(line)
        if train_match and ".trainer" in line:
            last_train_iter = int(train_match.group("iter"))

        val_match = VAL_RE.search(line)
        if val_match:
            flush_current()
            if val_match.group("iter"):
                iteration = int(val_match.group("iter"))
                iteration_source = "explicit_log"
            else:
                iteration, iteration_source = _infer_eval_iter(
                    last_train_iter,
                    meta.get("checkpoint_period"),
                    meta.get("max_iter"),
                    meta.get("stop_iter"),
                    len(rows) + 1,
                )
            current = {
                "iteration": iteration,
                "method": meta["method"],
                "checkpoint": "",
                "mIoU": _float(val_match.group("miou")),
                "mAcc": _float(val_match.group("macc")),
                "allAcc": _float(val_match.group("allacc")),
                "classes": [],
                "metric_source": meta["log_path"],
                "iteration_source": iteration_source,
            }
            continue

        class_match = CLASS_RE.search(line)
        if class_match and current is not None:
            current["classes"].append(
                {
                    "id": int(class_match.group("id")),
                    "name": class_match.group("name").strip(),
                    "iou": _float(class_match.group("iou")),
                    "accuracy": _float(class_match.group("acc")),
                }
            )

    flush_current()
    return rows


def _dedupe_metrics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (
            row.get("iteration"),
            round(_float(row.get("mIoU")), 8),
            round(_float(row.get("mAcc")), 8),
            round(_float(row.get("allAcc")), 8),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _load_run_metrics(run_dir: Path, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_csv.exists():
        rows = _read_metrics_csv(metrics_csv)
    elif metrics_jsonl.exists():
        rows = _read_metrics_jsonl(metrics_jsonl)
    else:
        rows = _parse_log_metrics(text, meta)

    for row in rows:
        if not row.get("method"):
            row["method"] = meta["method"]
        row["run_id"] = meta["run_id"]
        row["setting"] = meta["setting"]
    rows.sort(key=lambda item: (_int(item.get("iteration"), 0) or 0, _float(item.get("mIoU"))))
    return _dedupe_metrics(rows)


def _checkpoint_filename(method: str, family: str, iteration: int, metric_checkpoint: str = "") -> str:
    if metric_checkpoint:
        return Path(metric_checkpoint).name
    if family == "remask":
        return "remask_model_iter{:06d}.pth".format(iteration)
    if family == "sfda_baseline":
        return "{}_model_iter{:06d}.pth".format(method, iteration)
    return "model_iter{:06d}.pth".format(iteration)


def _checkpoint_paths(meta: Dict[str, Any], best: Dict[str, Any]) -> Dict[str, str]:
    filename = _checkpoint_filename(
        meta["method"],
        meta["family"],
        int(best["iteration"]),
        str(best.get("checkpoint", "")),
    )
    local_path = str(Path(meta["run_dir"]) / filename)
    logged_checkpoint = str(best.get("checkpoint") or "")
    if not logged_checkpoint and meta.get("output_dir_logged"):
        logged_checkpoint = str(Path(meta["output_dir_logged"]) / filename)
    return {
        "checkpoint_filename": filename,
        "local_checkpoint": local_path,
        "logged_checkpoint": logged_checkpoint,
        "checkpoint_exists_local": Path(local_path).exists(),
    }


def _mean(values: Iterable[float]) -> float:
    vals = [value for value in values if not math.isnan(value)]
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))


def _class_value(row: Dict[str, Any], class_id: int, key: str = "iou") -> float:
    for item in row.get("classes", []):
        if int(item.get("id", -1)) == class_id:
            return _float(item.get(key))
    return math.nan


def _rare_class_ids(row: Dict[str, Any], override_ids: Sequence[int]) -> List[int]:
    if override_ids:
        return list(override_ids)
    rare_tokens = ("human", "diver", "wreck", "robot", "instrument", "fish")
    ids = []
    for item in row.get("classes", []):
        name = str(item.get("name", "")).lower()
        if any(token in name for token in rare_tokens):
            ids.append(int(item.get("id", -1)))
    return [item for item in ids if item >= 0]


def _foreground_ids(row: Dict[str, Any]) -> List[int]:
    ids = []
    for item in row.get("classes", []):
        name = str(item.get("name", "")).lower()
        class_id = int(item.get("id", -1))
        if class_id < 0:
            continue
        if class_id == 0 or "background" in name:
            continue
        if "sea-floor" in name or "seafloor" in name:
            continue
        ids.append(class_id)
    return ids


def _pct(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return ""
    return "{:.2f}".format(float(value) * 100.0)


def _raw(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return ""
    return "{:.6f}".format(float(value))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in str_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"

    lines = [fmt([str(header) for header in headers])]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flatten_curve_row(meta: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "run_id": meta["run_id"],
        "setting": meta["setting"],
        "method": meta["method"],
        "family": meta["family"],
        "iteration": row["iteration"],
        "mIoU": row["mIoU"],
        "mAcc": row["mAcc"],
        "allAcc": row["allAcc"],
        "metric_source": row.get("metric_source", ""),
        "iteration_source": row.get("iteration_source", ""),
        "checkpoint": row.get("checkpoint", ""),
    }
    for item in row.get("classes", []):
        class_id = int(item.get("id", -1))
        if class_id < 0:
            continue
        flat["class_{}_name".format(class_id)] = item.get("name", "")
        flat["class_{}_iou".format(class_id)] = item.get("iou", "")
        flat["class_{}_acc".format(class_id)] = item.get("accuracy", "")
    return flat


def _sort_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    method = str(row.get("method", ""))
    try:
        order = METHOD_ORDER.index(method)
    except ValueError:
        order = len(METHOD_ORDER)
    return str(row.get("setting", "")), order, method


def _collect_runs(results_root: Path, rare_ids: Sequence[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifests: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    run_payloads: List[Dict[str, Any]] = []

    for log_path in sorted(results_root.rglob("log.txt")):
        run_dir = log_path.parent
        text = _read_text(log_path)
        meta = _parse_run_metadata(results_root, run_dir, text)
        metrics = _load_run_metrics(run_dir, text, meta)
        if not metrics:
            run_payloads.append({"meta": meta, "metrics": [], "best": None, "final": None})
            continue

        best = max(metrics, key=lambda row: _float(row.get("mIoU"), -math.inf))
        final = max(metrics, key=lambda row: _int(row.get("iteration"), -1) or -1)
        checkpoint_info = _checkpoint_paths(meta, best)
        rare_class_ids = _rare_class_ids(best, rare_ids)
        foreground_ids = _foreground_ids(best)
        rare_miou = _mean(_class_value(best, class_id) for class_id in rare_class_ids)
        foreground_miou = _mean(_class_value(best, class_id) for class_id in foreground_ids)

        manifest = {
            **{key: value for key, value in meta.items() if key != "opts"},
            "opts_json": json.dumps(meta.get("opts", []), ensure_ascii=False),
            "best_iteration": best["iteration"],
            "best_mIoU": best["mIoU"],
            "best_mAcc": best["mAcc"],
            "best_allAcc": best["allAcc"],
            "final_iteration": final["iteration"],
            "final_mIoU": final["mIoU"],
            "final_mAcc": final["mAcc"],
            "final_allAcc": final["allAcc"],
            "best_minus_final_mIoU": _float(best["mIoU"]) - _float(final["mIoU"]),
            "num_evals": len(metrics),
            "rare_class_ids": ",".join(str(item) for item in rare_class_ids),
            "rare_mIoU": rare_miou,
            "foreground_class_ids": ",".join(str(item) for item in foreground_ids),
            "foreground_mIoU": foreground_miou,
            "metric_source": best.get("metric_source", ""),
            "iteration_source": best.get("iteration_source", ""),
            **checkpoint_info,
            "classes": best.get("classes", []),
        }
        manifests.append(manifest)
        run_payloads.append({"meta": meta, "metrics": metrics, "best": best, "final": final})
        curves.extend(_flatten_curve_row(meta, row) for row in metrics)

    manifests.sort(key=_sort_key)
    curves.sort(key=lambda row: (row.get("setting", ""), row.get("method", ""), _int(row.get("iteration"), 0) or 0))
    return manifests, curves, run_payloads


def _main_table_rows(manifests: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int]]:
    class_ids = sorted(
        {
            int(item.get("id", -1))
            for row in manifests
            for item in row.get("classes", [])
            if int(item.get("id", -1)) >= 0
        }
    )
    rows = []
    for row in manifests:
        flat = {
            "setting": row["setting"],
            "method": row["method"],
            "best_iteration": row["best_iteration"],
            "mIoU": row["best_mIoU"],
            "mIoU_pct": float(row["best_mIoU"]) * 100.0,
            "rare_mIoU": row.get("rare_mIoU", math.nan),
            "rare_mIoU_pct": float(row.get("rare_mIoU", math.nan)) * 100.0
            if not math.isnan(float(row.get("rare_mIoU", math.nan)))
            else math.nan,
            "checkpoint": row["logged_checkpoint"] or row["local_checkpoint"],
        }
        class_by_id = {int(item["id"]): item for item in row.get("classes", [])}
        for class_id in class_ids:
            item = class_by_id.get(class_id, {})
            flat["class_{}_name".format(class_id)] = item.get("name", str(class_id))
            flat["class_{}_iou".format(class_id)] = item.get("iou", "")
            flat["class_{}_iou_pct".format(class_id)] = (
                float(item.get("iou")) * 100.0 if item.get("iou") not in (None, "") else ""
            )
        rows.append(flat)
    return rows, class_ids


def _training_summary(manifests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in manifests:
        best = _float(row.get("best_mIoU"))
        final = _float(row.get("final_mIoU"))
        rows.append(
            {
                "setting": row["setting"],
                "method": row["method"],
                "best_iteration": row["best_iteration"],
                "final_iteration": row["final_iteration"],
                "best_mIoU": best,
                "final_mIoU": final,
                "late_drop_from_best": best - final,
                "num_evals": row["num_evals"],
                "iteration_source": row["iteration_source"],
            }
        )
    return rows


def _cross_dataset_rows(manifests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in manifests:
        grouped[row["method"]][row["setting"]] = _float(row["best_mIoU"])

    settings = sorted({row["setting"] for row in manifests})
    rows = []
    for method in sorted(grouped, key=lambda item: METHOD_ORDER.index(item) if item in METHOD_ORDER else 999):
        values = [grouped[method].get(setting, math.nan) for setting in settings]
        present = [value for value in values if not math.isnan(value)]
        flat: Dict[str, Any] = {"method": method}
        for setting, value in zip(settings, values):
            flat[setting] = value
        flat["avg_mIoU"] = _mean(present)
        flat["worst_mIoU"] = min(present) if present else math.nan
        flat["num_settings"] = len(present)
        rows.append(flat)
    return rows


def _rare_rows(manifests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in manifests:
        rows.append(
            {
                "setting": row["setting"],
                "method": row["method"],
                "best_iteration": row["best_iteration"],
                "mIoU": row["best_mIoU"],
                "foreground_mIoU": row.get("foreground_mIoU", math.nan),
                "rare_mIoU": row.get("rare_mIoU", math.nan),
                "rare_class_ids": row.get("rare_class_ids", ""),
                "foreground_class_ids": row.get("foreground_class_ids", ""),
            }
        )
    return rows


def _write_main_markdown(path: Path, table_rows: Sequence[Dict[str, Any]], class_ids: Sequence[int]) -> None:
    if not table_rows:
        path.write_text("No metrics found.\n", encoding="utf-8")
        return
    class_labels = {}
    for class_id in class_ids:
        for row in table_rows:
            label = row.get("class_{}_name".format(class_id))
            if label:
                class_labels[class_id] = "{}:{}".format(class_id, label)
                break
        class_labels.setdefault(class_id, "class_{}".format(class_id))

    headers = ["Setting", "Method", "Best iter"] + [class_labels[class_id] for class_id in class_ids]
    headers += ["mIoU", "rare-mIoU"]
    rows = []
    for row in table_rows:
        rows.append(
            [row["setting"], row["method"], row["best_iteration"]]
            + [_pct(_float(row.get("class_{}_iou".format(class_id)))) for class_id in class_ids]
            + [_pct(_float(row.get("mIoU"))), _pct(_float(row.get("rare_mIoU")))]
        )
    path.write_text(_markdown_table(headers, rows) + "\n", encoding="utf-8")


def _run_evidence_section(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    best = payload.get("best")
    final = payload.get("final")
    if best is None:
        return "### {}\n\nNo validation metrics were found in this run.\n".format(meta["run_id"])

    class_rows = [
        [item.get("id", ""), item.get("name", ""), _pct(_float(item.get("iou"))), _pct(_float(item.get("accuracy")))]
        for item in best.get("classes", [])
    ]
    checkpoint_info = _checkpoint_paths(meta, best)
    lines = [
        "### {}".format(meta["run_id"]),
        "",
        "- Method: `{}`".format(meta["method"]),
        "- Setting: `{}`".format(meta["setting"]),
        "- Config: `{}`".format(meta.get("config_file", "")),
        "- Best validation: iter `{}` with mIoU `{}`%, mAcc `{}`%, allAcc `{}`%".format(
            best["iteration"],
            _pct(_float(best["mIoU"])),
            _pct(_float(best["mAcc"])),
            _pct(_float(best["allAcc"])),
        ),
        "- Final validation: iter `{}` with mIoU `{}`%".format(
            final["iteration"] if final else "",
            _pct(_float(final["mIoU"])) if final else "",
        ),
        "- Expected checkpoint: `{}`".format(
            checkpoint_info["logged_checkpoint"] or checkpoint_info["local_checkpoint"]
        ),
        "- Local checkpoint present: `{}`".format(str(checkpoint_info["checkpoint_exists_local"])),
        "",
        _markdown_table(["Class id", "Class", "IoU (%)", "Acc (%)"], class_rows),
        "",
    ]
    return "\n".join(lines)


def _write_evidence_markdown(
    path: Path,
    manifests: Sequence[Dict[str, Any]],
    main_table_md: Path,
    run_payloads: Sequence[Dict[str, Any]],
) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_rows = [
        [
            row["setting"],
            row["method"],
            row["best_iteration"],
            _pct(_float(row["best_mIoU"])),
            _pct(_float(row.get("rare_mIoU", math.nan))),
            row["checkpoint_filename"],
        ]
        for row in manifests
    ]
    training_rows = [
        [
            row["setting"],
            row["method"],
            row["best_iteration"],
            row["final_iteration"],
            _pct(_float(row["best_mIoU"])),
            _pct(_float(row["final_mIoU"])),
            _pct(_float(row["best_minus_final_mIoU"])),
        ]
        for row in manifests
    ]
    status_rows = [
        ["Main benchmark table", "Generated", "Per-class IoU, mIoU, rare-mIoU from log/metrics files."],
        ["Cross-dataset matrix", "Generated when multiple settings exist", "Avg and worst mIoU across settings."],
        ["Training curve", "Generated", "All validation points exported to metrics_curves.csv."],
        ["Rare/small-object analysis", "Generated", "Uses default underwater object classes unless overridden."],
        ["Best checkpoint loading", "Manifest generated", "Run eval_underwater_best_iters.py on HPC where .pth files exist."],
        ["Qualitative inference", "Requires predictions", "Use re-evaluated checkpoints to export prediction masks/figures."],
        ["Pseudo-label quality", "Requires pseudo-label artifacts", "Not present in current metric logs."],
        ["Stability/uncertainty maps", "Requires saved maps or extra evaluation", "Not derivable from scalar log metrics alone."],
        ["Quality-stratified/boundary metrics", "Requires image-level predictions", "Can be added after best-checkpoint re-evaluation."],
    ]

    lines = [
        "# Underwater SFUDA Evidence Blocks",
        "",
        "Generated at `{}`.".format(generated),
        "",
        "Validation-label results here are for reporting and analysis. They should not be used to tune SFUDA training.",
        "",
        "## 1. Best-iteration manifest",
        "",
        _markdown_table(
            ["Setting", "Method", "Best iter", "mIoU (%)", "rare-mIoU (%)", "Checkpoint file"],
            best_rows,
        ),
        "",
        "## 2. Main benchmark table",
        "",
        "The full Markdown table is saved at `{}`.".format(main_table_md.as_posix()),
        "",
        main_table_md.read_text(encoding="utf-8") if main_table_md.exists() else "",
        "",
        "## 3. Training-curve evidence",
        "",
        _markdown_table(
            ["Setting", "Method", "Best iter", "Final iter", "Best mIoU (%)", "Final mIoU (%)", "Best-final gap (%)"],
            training_rows,
        ),
        "",
        "## 4. Evidence block coverage",
        "",
        _markdown_table(["Evidence block", "Status", "Note"], status_rows),
        "",
        "## 5. Per-run evidence blocks",
        "",
    ]
    for payload in sorted(run_payloads, key=lambda item: _sort_key(item["meta"])):
        lines.append(_run_evidence_section(payload))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract best validation mIoU/iteration and paper-style evidence blocks from underwater results."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results_underwater"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_underwater") / "evidence_summary")
    parser.add_argument(
        "--rare-class-ids",
        default="",
        help="Comma-separated override for rare-class mIoU. Default detects human/wreck/robot/fish classes by name.",
    )
    args = parser.parse_args()

    results_root = args.results_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests, curves, run_payloads = _collect_runs(results_root, _parse_int_list(args.rare_class_ids))
    main_rows, class_ids = _main_table_rows(manifests)
    training_rows = _training_summary(manifests)
    cross_rows = _cross_dataset_rows(manifests)
    rare_rows = _rare_rows(manifests)

    manifest_csv_rows = []
    manifest_json_rows = []
    for row in manifests:
        csv_row = {key: value for key, value in row.items() if key != "classes"}
        manifest_csv_rows.append(csv_row)
        manifest_json_rows.append(row)

    manifest_fields = [
        "run_id",
        "setting",
        "method",
        "family",
        "config_file",
        "source_train",
        "target_train",
        "test_set",
        "output_dir_logged",
        "checkpoint_period",
        "max_iter",
        "stop_iter",
        "best_iteration",
        "best_mIoU",
        "best_mAcc",
        "best_allAcc",
        "final_iteration",
        "final_mIoU",
        "best_minus_final_mIoU",
        "rare_mIoU",
        "rare_class_ids",
        "foreground_mIoU",
        "foreground_class_ids",
        "num_evals",
        "checkpoint_filename",
        "local_checkpoint",
        "logged_checkpoint",
        "checkpoint_exists_local",
        "metric_source",
        "iteration_source",
        "run_dir",
        "log_path",
        "seafloor_ignore",
        "suim_legacy_color_map",
        "paper_preset",
        "opts_json",
    ]
    _write_csv(output_dir / "best_iters_manifest.csv", manifest_csv_rows, manifest_fields)
    (output_dir / "best_iters_manifest.json").write_text(
        json.dumps(manifest_json_rows, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    curve_fields = sorted({key for row in curves for key in row.keys()})
    _write_csv(output_dir / "metrics_curves.csv", curves, curve_fields)

    main_fields = sorted({key for row in main_rows for key in row.keys()})
    _write_csv(output_dir / "main_benchmark_table.csv", main_rows, main_fields)
    main_md = output_dir / "main_benchmark_table.md"
    _write_main_markdown(main_md, main_rows, class_ids)

    _write_csv(
        output_dir / "training_curve_summary.csv",
        training_rows,
        [
            "setting",
            "method",
            "best_iteration",
            "final_iteration",
            "best_mIoU",
            "final_mIoU",
            "late_drop_from_best",
            "num_evals",
            "iteration_source",
        ],
    )

    cross_fields = ["method"] + sorted({key for row in cross_rows for key in row if key not in {"method", "avg_mIoU", "worst_mIoU", "num_settings"}})
    cross_fields += ["avg_mIoU", "worst_mIoU", "num_settings"]
    _write_csv(output_dir / "cross_dataset_matrix.csv", cross_rows, cross_fields)

    _write_csv(
        output_dir / "rare_class_table.csv",
        rare_rows,
        [
            "setting",
            "method",
            "best_iteration",
            "mIoU",
            "foreground_mIoU",
            "rare_mIoU",
            "rare_class_ids",
            "foreground_class_ids",
        ],
    )

    _write_evidence_markdown(output_dir / "evidence_blocks.md", manifests, main_md, run_payloads)

    print("Found {} runs with validation metrics.".format(len(manifests)))
    print("Wrote {}".format(output_dir / "best_iters_manifest.csv"))
    print("Wrote {}".format(output_dir / "evidence_blocks.md"))
    if manifests:
        headers = ["setting", "method", "best_iter", "mIoU(%)", "checkpoint"]
        preview = [
            [
                row["setting"],
                row["method"],
                row["best_iteration"],
                _pct(_float(row["best_mIoU"])),
                row["checkpoint_filename"],
            ]
            for row in manifests[:20]
        ]
        print(_markdown_table(headers, preview))


if __name__ == "__main__":
    main()
