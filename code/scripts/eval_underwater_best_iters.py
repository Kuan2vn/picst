#!/usr/bin/env python3
"""Evaluate best-iteration underwater checkpoints from a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("classes"):
                try:
                    row["classes"] = json.loads(row["classes"])
                except json.JSONDecodeError:
                    row["classes"] = []
            rows.append(row)
    return rows


def _safe_run_id(row: Dict[str, Any]) -> str:
    raw = row.get("run_id") or "{}_{}".format(row.get("setting", "run"), row.get("method", "method"))
    safe = str(raw).replace("\\", "/").strip("/")
    for ch in '/:*?"<>| ':
        safe = safe.replace(ch, "_")
    return safe


def _candidate_paths(row: Dict[str, Any], checkpoint_root: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []
    for key in ("local_checkpoint", "logged_checkpoint"):
        value = row.get(key)
        if value:
            candidates.append(Path(str(value)))
    filename = row.get("checkpoint_filename")
    if filename:
        run_dir = row.get("run_dir")
        if run_dir:
            candidates.append(Path(str(run_dir)) / str(filename))
        logged_output = row.get("output_dir_logged")
        if logged_output:
            candidates.append(Path(str(logged_output)) / str(filename))
    if checkpoint_root is not None:
        rooted = []
        for path in candidates:
            if path.is_absolute():
                rooted.append(path)
            else:
                rooted.append(checkpoint_root / path)
        candidates = rooted + candidates

    unique: List[Path] = []
    seen = set()
    for path in candidates:
        norm = str(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def _resolve_checkpoint(row: Dict[str, Any], checkpoint_root: Optional[Path]) -> Optional[Path]:
    for path in _candidate_paths(row, checkpoint_root):
        if path.exists():
            return path
    return None


def _opts_without(opts: Sequence[Any], keys: Iterable[str]) -> List[str]:
    skip = set(keys)
    cleaned: List[str] = []
    index = 0
    while index < len(opts):
        key = str(opts[index])
        if key in skip and index + 1 < len(opts):
            index += 2
            continue
        cleaned.append(key)
        index += 1
    return cleaned


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _load_cfg(row: Dict[str, Any], checkpoint_path: Path, output_dir: Path, device: str, batch_size: Optional[int]):
    from core.configs.defaults import _C as default_cfg
    from core.configs.underwater import apply_underwater_label_args

    opts = json.loads(row.get("opts_json") or "[]")
    opts = _opts_without(opts, {"resume", "OUTPUT_DIR"})

    cfg = default_cfg.clone()
    config_file = row.get("config_file")
    if not config_file:
        raise ValueError("Missing config_file in manifest row {}".format(row.get("run_id", "")))
    cfg.merge_from_file(str(config_file))
    if opts:
        cfg.merge_from_list([str(item) for item in opts])
    args = SimpleNamespace(
        seafloor_ignore=_to_bool(row.get("seafloor_ignore", False)),
        suim_legacy_color_map=_to_bool(row.get("suim_legacy_color_map", False)),
    )
    apply_underwater_label_args(cfg, args)
    cfg.resume = str(checkpoint_path)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.MODEL.DEVICE = device
    if batch_size is not None:
        cfg.TEST.BATCH_SIZE = int(batch_size)
    cfg.freeze()
    return cfg


def _class_name(dataset: Any, index: int) -> str:
    names = getattr(dataset, "trainid2name", None)
    if isinstance(names, dict):
        return str(names.get(index, index))
    if isinstance(names, (list, tuple)) and index < len(names):
        return str(names[index])
    return str(index)


def _intersection_and_union(pred, target, num_classes: int, ignore_label: int):
    import torch

    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = target != ignore_label
    pred = pred[valid]
    target = target[valid]
    intersection = pred[pred == target]
    area_intersection = torch.bincount(intersection, minlength=num_classes)[:num_classes].double()
    area_output = torch.bincount(pred, minlength=num_classes)[:num_classes].double()
    area_target = torch.bincount(target, minlength=num_classes)[:num_classes].double()
    area_union = area_output + area_target - area_intersection
    return area_intersection.cpu(), area_union.cpu(), area_target.cpu()


def _load_model(cfg, checkpoint_path: Path):
    import torch

    from core.models import build_classifier, build_feature_extractor
    from core.utils.misc import strip_prefix_if_present

    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor = build_feature_extractor(cfg).to(device)
    classifier = build_classifier(cfg).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    if "feature_extractor" not in checkpoint or "classifier" not in checkpoint:
        raise KeyError(
            "Checkpoint {} does not contain feature_extractor/classifier keys. Keys: {}".format(
                checkpoint_path,
                sorted(checkpoint.keys()),
            )
        )
    feature_weights = strip_prefix_if_present(checkpoint["feature_extractor"], "module.")
    classifier_weights = strip_prefix_if_present(checkpoint["classifier"], "module.")
    feature_extractor.load_state_dict(feature_weights, strict=False)
    classifier.load_state_dict(classifier_weights, strict=False)
    return feature_extractor, classifier, checkpoint


def _evaluate(cfg, feature_extractor, classifier, method: str, checkpoint_path: Path, iteration: int) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from core.datasets import build_dataset

    device = torch.device(cfg.MODEL.DEVICE)
    test_data = build_dataset(cfg, mode="test", is_source=False)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    feature_extractor.eval()
    classifier.eval()
    total_intersection = None
    total_union = None
    total_target = None
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True).long()
            _, features = feature_extractor(x)
            logits = classifier(features, y.shape[-2:])
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=True)
            pred = logits.max(1)[1]
            inter, union, target = _intersection_and_union(
                pred,
                y,
                cfg.MODEL.NUM_CLASSES,
                cfg.INPUT.IGNORE_LABEL,
            )
            if total_intersection is None:
                total_intersection = inter
                total_union = union
                total_target = target
            else:
                total_intersection += inter
                total_union += union
                total_target += target

    if total_intersection is None or total_union is None or total_target is None:
        raise RuntimeError("No test batches were produced for {}".format(cfg.DATASETS.TEST))

    iou_class = total_intersection.numpy() / (total_union.numpy() + 1e-10)
    accuracy_class = total_intersection.numpy() / (total_target.numpy() + 1e-10)
    m_iou = float(iou_class.mean())
    m_acc = float(accuracy_class.mean())
    all_acc = float(total_intersection.sum().item() / (total_target.sum().item() + 1e-10))

    classes = []
    for index in range(cfg.MODEL.NUM_CLASSES):
        classes.append(
            {
                "id": int(index),
                "name": _class_name(test_data, index),
                "iou": float(iou_class[index]),
                "accuracy": float(accuracy_class[index]),
            }
        )

    return {
        "iteration": int(iteration),
        "method": method,
        "checkpoint": str(checkpoint_path),
        "dataset": str(cfg.DATASETS.TEST),
        "mIoU": m_iou,
        "mAcc": m_acc,
        "allAcc": all_acc,
        "classes": classes,
    }


def _flatten(metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "iteration": metrics["iteration"],
        "method": metrics["method"],
        "dataset": metrics.get("dataset", ""),
        "checkpoint": metrics.get("checkpoint", ""),
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "allAcc": metrics["allAcc"],
    }
    for item in metrics.get("classes", []):
        idx = int(item["id"])
        row["class_{}_name".format(idx)] = item.get("name", "")
        row["class_{}_iou".format(idx)] = item.get("iou", "")
        row["class_{}_acc".format(idx)] = item.get("accuracy", "")
    return row


def _write_metrics(out_dir: Path, metrics: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    flat = _flatten(metrics)
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    class_rows = [
        [item["id"], item["name"], "{:.2f}".format(item["iou"] * 100.0), "{:.2f}".format(item["accuracy"] * 100.0)]
        for item in metrics.get("classes", [])
    ]
    lines = [
        "# Best-checkpoint evaluation",
        "",
        "- Method: `{}`".format(metrics["method"]),
        "- Iteration: `{}`".format(metrics["iteration"]),
        "- Dataset: `{}`".format(metrics.get("dataset", "")),
        "- Checkpoint: `{}`".format(metrics.get("checkpoint", "")),
        "- mIoU/mAcc/allAcc: `{:.2f}/{:.2f}/{:.2f}`".format(
            metrics["mIoU"] * 100.0,
            metrics["mAcc"] * 100.0,
            metrics["allAcc"] * 100.0,
        ),
        "",
        _markdown_table(["Class id", "Class", "IoU (%)", "Acc (%)"], class_rows),
        "",
    ]
    (out_dir / "evidence_block.md").write_text("\n".join(lines), encoding="utf-8")


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


def _select_rows(rows: Sequence[Dict[str, Any]], row_index: Optional[int], all_rows: bool, method: Optional[str]) -> List[Dict[str, Any]]:
    selected = list(rows)
    if method:
        selected = [row for row in selected if row.get("method") == method]
    if row_index is not None:
        if row_index < 0 or row_index >= len(selected):
            raise IndexError("row-index {} out of range for {} selected rows".format(row_index, len(selected)))
        return [selected[row_index]]
    if all_rows:
        return selected
    return selected[:1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best-iteration checkpoints from best_iters_manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("results_underwater") / "evidence_summary" / "best_iters_manifest.json")
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results_underwater") / "evidence_summary" / "best_eval")
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Evaluate all selected rows in this process.")
    parser.add_argument("--method", default=None, help="Optional method filter before row-index selection.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if a selected checkpoint is missing.")
    args = parser.parse_args()

    rows = _load_manifest(args.manifest)
    selected = _select_rows(rows, args.row_index, args.all, args.method)
    if not selected:
        print("No rows selected from {}".format(args.manifest))
        return

    aggregate: List[Dict[str, Any]] = []
    for row in selected:
        checkpoint = _resolve_checkpoint(row, args.checkpoint_root)
        run_id = _safe_run_id(row)
        out_dir = args.output_dir / run_id
        if checkpoint is None:
            message = "Missing checkpoint for {} / {}. Tried: {}".format(
                row.get("setting", ""),
                row.get("method", ""),
                ", ".join(str(path) for path in _candidate_paths(row, args.checkpoint_root)),
            )
            if args.strict:
                raise FileNotFoundError(message)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "missing_checkpoint.txt").write_text(message + "\n", encoding="utf-8")
            print(message, file=sys.stderr)
            continue

        cfg = _load_cfg(row, checkpoint, out_dir, args.device, args.batch_size)
        feature_extractor, classifier, checkpoint_obj = _load_model(cfg, checkpoint)
        iteration = int(float(row.get("best_iteration") or checkpoint_obj.get("iteration", 0)))
        metrics = _evaluate(cfg, feature_extractor, classifier, row.get("method", "method"), checkpoint, iteration)
        _write_metrics(out_dir, metrics)
        aggregate.append(_flatten(metrics))
        print(
            "Evaluated {} at iter {}: mIoU {:.4f}".format(
                row.get("run_id", ""),
                iteration,
                metrics["mIoU"],
            )
        )

    if aggregate and len(aggregate) > 1:
        fields = sorted({key for row in aggregate for key in row.keys()})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(aggregate)


if __name__ == "__main__":
    main()
