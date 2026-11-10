#!/usr/bin/env python3
"""Summarize the best classwise validation result from each HPC .out log."""

from __future__ import annotations

import argparse
import ast
import csv
import re
import unicodedata
from pathlib import Path
from typing import Any


VAL_RE = re.compile(
    r"Val result: mIoU/mAcc/allAcc "
    r"(?P<miou>\d+\.\d+)/(?P<macc>\d+\.\d+)/(?P<allacc>\d+\.\d+)"
)
CLASS_RE = re.compile(
    r"Class_(?P<index>\d+) (?P<name>.+?) Result: iou/accuracy "
    r"(?P<iou>\d+\.\d+)/(?P<accuracy>\d+\.\d+)"
)
BEST_RE = re.compile(r"Early-stop monitor: new best mIoU \d+\.\d+ at iter (?P<iter>\d+)")
JOB_RE = re.compile(r"_(?P<job>\d+)_(?P<task>\d+)\.out$")

FIELDS_BEFORE_CLASSES = [
    "run_id",
    "method",
    "domain",
    "source_train",
    "target_train",
    "test_set",
    "reliability_mode",
    "fuse_mode",
    "class_balance",
    "class_balance_gamma",
    "class_balance_min",
    "class_balance_max",
    "contrastive",
    "contrastive_lambda",
    "uwrg",
    "uwrg_mode",
    "uwrg_composition",
    "uwrg_floor",
    "uwrg_lambda",
    "uwrg_strength",
    "uwrg_warmup_ratio",
    "uwrg_segment_stride",
    "uwrg_min_segment_pixels",
    "tcr_update_frequency",
    "history_update_mode",
    "history_update_period",
    "history_momentum",
    "temporal_floor",
    "base_lr",
    "batch_size",
    "max_iter",
    "checkpoint_period",
    "seed",
    "init_checkpoint",
    "output_dir",
    "best_iter",
    "mIoU",
    "mAcc",
    "allAcc",
]

FIELDS_AFTER_CLASSES = [
    "num_classes",
    "classwise_mean_check",
    "final_mIoU",
    "num_validation_blocks",
    "run_completed",
    "job_id",
    "array_task_id",
    "log_file",
]


def parse_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def value_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def scalar(namespace: str, key: str, default: Any = "") -> Any:
    match = re.search(rf"(?:^|[,( ]){re.escape(key)}=([^,)]*)", namespace)
    if not match:
        return default
    raw = match.group(1).strip().strip("'")
    if raw in {"True", "False"}:
        return raw == "True"
    return raw


def parse_options(namespace: str) -> tuple[str, dict[str, str]]:
    match = re.search(r"opts=(\[.*\]), distributed=", namespace)
    if not match:
        return "", {}
    tokens = ast.literal_eval(match.group(1))
    init_checkpoint = ""
    options: dict[str, str] = {}
    index = 0
    if len(tokens) >= 2 and tokens[0] == "resume":
        init_checkpoint = str(tokens[1])
        index = 2
    while index + 1 < len(tokens):
        options[str(tokens[index])] = str(tokens[index + 1])
        index += 2
    return init_checkpoint, options


def option(options: dict[str, str], key: str, default: Any = "") -> Any:
    return options.get(key, default)


def domain_name(source_train: str, target_train: str) -> str:
    source_names = {
        "uiis_train": "UIIS",
        "usis10k_bg_train": "USIS10K-bg",
        "gta5_train": "GTA5",
    }
    target_names = {
        "suim_train": "SUIM",
        "cityscapes_train": "Cityscapes",
    }
    source = source_names.get(source_train, source_train or "unknown")
    target = target_names.get(target_train, target_train or "unknown")
    return f"{source}->{target}"


def short_init(checkpoint: str) -> str:
    path = checkpoint.replace("\\", "/")
    iteration_match = re.search(r"(?:model|remask_model)_iter(\d+)\.pth$", path)
    iteration = f"@{iteration_match.group(1)}" if iteration_match else ""
    if "source_model/" in path:
        return f"source{iteration}"
    if "control_disable_uwrg/" in path:
        return f"entropyKL-noUWRG{iteration}"
    if "remask_special_uwrg_sensitivity_" in path:
        variant = Path(path).parent.name
        return f"entropyKL-UWRG-{variant}{iteration}"
    return f"{Path(path).stem}" if path else "unknown"


def build_method(row: dict[str, Any]) -> str:
    parts = [row["domain"], f"rel={row['reliability_mode']}"]
    if row["class_balance"]:
        parts.append(
            "CB("
            f"gamma={row['class_balance_gamma']},"
            f"range={row['class_balance_min']}-{row['class_balance_max']}"
            ")"
        )
    else:
        parts.append("CB=off")

    if row["contrastive"]:
        parts.append(f"Con(lam={row['contrastive_lambda']})")
    else:
        parts.append("Con=off")

    if row["uwrg"]:
        parts.append(
            "UWRG("
            f"{row['uwrg_mode']},floor={row['uwrg_floor']},"
            f"lam={row['uwrg_lambda']},strength={row['uwrg_strength']},"
            f"warmup={row['uwrg_warmup_ratio']},"
            f"stride={row['uwrg_segment_stride']},"
            f"minpix={row['uwrg_min_segment_pixels']}"
            ")"
        )
    else:
        parts.append("UWRG=off")

    tcr_frequency = str(row["tcr_update_frequency"])
    parts.append(f"TCR={'off' if tcr_frequency == '999999' else tcr_frequency}")
    parts.append(f"init={short_init(row['init_checkpoint'])}")
    return " | ".join(parts)


def parse_metric_blocks(lines: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        if "DTST.tester INFO: Val result:" not in line:
            continue
        val_match = VAL_RE.search(line)
        if not val_match:
            continue
        block: dict[str, Any] = {
            "mIoU": float(val_match.group("miou")),
            "mAcc": float(val_match.group("macc")),
            "allAcc": float(val_match.group("allacc")),
            "best_iter": "",
            "classes": {},
        }
        for following in lines[line_index + 1 : line_index + 40]:
            class_match = CLASS_RE.search(following)
            if class_match:
                block["classes"][int(class_match.group("index"))] = {
                    "name": class_match.group("name"),
                    "iou": float(class_match.group("iou")),
                }
            best_match = BEST_RE.search(following)
            if best_match:
                block["best_iter"] = int(best_match.group("iter"))
                break
            if VAL_RE.search(following):
                break
        blocks.append(block)
    return blocks


def class_column(index: int, name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return f"iou_{index:02d}_{slug}"


def parse_log(log_path: Path, root: Path) -> dict[str, Any] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    namespace_line = next((line for line in lines if "DTST INFO: Namespace(" in line), "")
    if not namespace_line:
        return None
    namespace = namespace_line.split("Namespace(", 1)[1]
    init_checkpoint, options = parse_options(namespace)
    blocks = parse_metric_blocks(lines)
    expected_classes = max((len(block["classes"]) for block in blocks), default=0)
    complete_blocks = [
        block for block in blocks if len(block["classes"]) == expected_classes
    ]
    if not complete_blocks:
        return None

    # The first occurrence of the maximum has the exact early-stop iteration.
    best = max(complete_blocks, key=lambda block: block["mIoU"])
    if best["best_iter"] == "":
        first_same = next(
            (
                block
                for block in complete_blocks
                if block["mIoU"] == best["mIoU"] and block["best_iter"] != ""
            ),
            None,
        )
        if first_same:
            best = first_same

    source_train = option(options, "DATASETS.SOURCE_TRAIN")
    target_train = option(options, "DATASETS.TARGET_TRAIN")
    job_match = JOB_RE.search(log_path.name)
    relative_log = log_path.relative_to(root).as_posix()
    run_id = log_path.relative_to(root).with_suffix("").as_posix().replace("/", "__")

    row: dict[str, Any] = {
        "run_id": run_id,
        "domain": domain_name(source_train, target_train),
        "source_train": source_train,
        "target_train": target_train,
        "test_set": option(options, "DATASETS.TEST"),
        "reliability_mode": option(options, "REMASK.RELIABILITY_MODE"),
        "fuse_mode": option(options, "REMASK.FUSE_MODE"),
        "class_balance": parse_bool(option(options, "REMASK.USE_CLASS_BALANCE")),
        "class_balance_gamma": option(options, "REMASK.CLASS_BALANCE_GAMMA"),
        "class_balance_min": option(options, "REMASK.CLASS_BALANCE_MIN"),
        "class_balance_max": option(options, "REMASK.CLASS_BALANCE_MAX"),
        "contrastive": parse_bool(option(options, "REMASK.USE_CONTRASTIVE")),
        "contrastive_lambda": option(options, "REMASK.LAMBDA"),
        "uwrg": bool(scalar(namespace, "use_uwrg", False)),
        "uwrg_mode": scalar(namespace, "uwrg_mode"),
        "uwrg_composition": scalar(namespace, "uwrg_composition"),
        "uwrg_floor": scalar(namespace, "uwrg_floor"),
        "uwrg_lambda": scalar(namespace, "uwrg_lambda"),
        "uwrg_strength": scalar(namespace, "uwrg_strength"),
        "uwrg_warmup_ratio": scalar(namespace, "uwrg_warmup_ratio"),
        "uwrg_segment_stride": scalar(namespace, "uwrg_segment_stride"),
        "uwrg_min_segment_pixels": scalar(namespace, "uwrg_min_segment_pixels"),
        "tcr_update_frequency": option(options, "TCR.UPDATE_FREQUENCY", "off"),
        "history_update_mode": option(options, "REMASK.HISTORY_UPDATE_MODE"),
        "history_update_period": option(options, "REMASK.HISTORY_UPDATE_PERIOD"),
        "history_momentum": option(options, "REMASK.HISTORY_MOMENTUM"),
        "temporal_floor": option(options, "REMASK.TEMPORAL_FLOOR"),
        "base_lr": option(options, "SOLVER.BASE_LR"),
        "batch_size": option(options, "SOLVER.BATCH_SIZE"),
        "max_iter": option(options, "SOLVER.MAX_ITER"),
        "checkpoint_period": option(options, "SOLVER.CHECKPOINT_PERIOD"),
        "seed": scalar(namespace, "seed"),
        "init_checkpoint": init_checkpoint,
        "output_dir": option(options, "OUTPUT_DIR"),
        "best_iter": best["best_iter"],
        "mIoU": best["mIoU"],
        "mAcc": best["mAcc"],
        "allAcc": best["allAcc"],
        "num_classes": expected_classes,
        "classwise_mean_check": round(
            sum(item["iou"] for item in best["classes"].values()) / expected_classes,
            4,
        ),
        "final_mIoU": complete_blocks[-1]["mIoU"],
        "num_validation_blocks": len(complete_blocks),
        "run_completed": "Total ReMask training time:" in text,
        "job_id": job_match.group("job") if job_match else "",
        "array_task_id": job_match.group("task") if job_match else "",
        "log_file": relative_log,
    }
    for class_index, item in best["classes"].items():
        row[class_column(class_index, item["name"])] = item["iou"]
    row["method"] = build_method(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("HPC_logs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HPC_logs/hpc_ablation_classwise_summary.csv"),
    )
    args = parser.parse_args()

    root = args.logs.resolve()
    rows = []
    skipped = []
    for log_path in sorted(root.rglob("*.out")):
        row = parse_log(log_path, root)
        if row is None:
            skipped.append(log_path.relative_to(root).as_posix())
        else:
            rows.append(row)

    rows.sort(key=lambda row: (row["domain"], row["method"], row["run_id"]))
    class_fields = sorted(
        {
            field
            for row in rows
            for field in row
            if field.startswith("iou_") and re.match(r"iou_\d\d_", field)
        },
        key=lambda field: (int(field.split("_", 2)[1]), field),
    )
    fields = [*FIELDS_BEFORE_CLASSES, *class_fields, *FIELDS_AFTER_CLASSES]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: value_text(row.get(field, "")) for field in fields})

    print(f"Wrote {len(rows)} runs to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} logs without complete classwise metrics:")
        for path in skipped:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
