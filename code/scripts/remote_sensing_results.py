"""Small, self-contained result reports; never include model/optimizer tensors."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path


MODEL_FIELDS = (
    "stage", "iteration", "inference_model", "num_classes", "seed",
    "train_policy", "normalization", "crop_size", "training_settings",
    "source_checkpoint_sha256", "evaluated_checkpoint_sha256",
    "architecture", "protocol", "protocol_sha256", "target_eval_manifest_sha256",
    "checkpoint_selection", "smoke_only", "method",
)


def compare_source(task: str, split: str, metrics: dict, model: dict,
                   baseline_path: Path | None) -> dict:
    if model.get("stage") == "source":
        return {"status": "not_applicable"}
    if baseline_path is None or not baseline_path.is_file():
        return {"status": "missing_source_evaluation"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("task") != task or baseline.get("split") != split:
        return {"status": "task_or_split_mismatch"}
    source_hash = model.get("source_checkpoint_sha256")
    if not source_hash or source_hash != baseline.get("source_checkpoint_sha256"):
        return {"status": "source_checkpoint_mismatch"}
    reference = baseline.get('model', baseline)
    for key in ('architecture', 'protocol', 'protocol_sha256', 'target_eval_manifest_sha256'):
        if model.get(key) != reference.get(key):
            return {'status': f'{key}_mismatch'}
    before = baseline["metrics"]
    if before.get("valid_pixels") != metrics.get("valid_pixels"):
        return {"status": "evaluation_pixel_count_mismatch"}
    if set(before.get("iou_by_class", {})) != set(metrics.get("iou_by_class", {})):
        return {"status": "class_mapping_mismatch"}
    deltas = {
        name: None if value is None or before["iou_by_class"][name] is None else
        100.0 * (value - before["iou_by_class"][name])
        for name, value in metrics["iou_by_class"].items()
    }
    result = {
        "status": "matched_source_checkpoint_and_evaluation_scope",
        "source_checkpoint_sha256": source_hash,
        "metrics": before,
        "delta_miou_percentage_points": 100.0 * (metrics["miou"] - before["miou"]),
        "delta_iou_by_class_percentage_points": deltas,
    }
    if "miou5" in before and "miou5" in metrics:
        result["delta_miou5_percentage_points"] = 100.0 * (metrics["miou5"] - before["miou5"])
    return result


def write_portable_result(path: Path, *, task: str, split: str, metrics: dict,
                          model: dict, protocol: dict,
                          training_protocol: dict | None = None,
                          dataset_metadata: dict | None = None,
                          source_baseline_path: Path | None = None) -> dict:
    # Explicit allowlist prevents a checkpoint's state_dict/optimizer from leaking.
    summary = {key: model[key] for key in MODEL_FIELDS if key in model}
    settings = summary.get("training_settings", {})
    method = summary.get('method') or ("source_only" if summary.get("stage") == "source" else (
        "pseudo_label_baseline" if settings.get("relation_weight") == 0 else
        "srmp_relation" if settings.get("relation_weight", 0) > 0 else "unknown"
    ))
    percentages = {"miou": 100.0 * metrics["miou"],
                   "iou_by_class": {key: None if value is None else 100.0 * value
                                    for key, value in metrics["iou_by_class"].items()}}
    if "miou5" in metrics:
        percentages["miou5"] = 100.0 * metrics["miou5"]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task, "split": split, "method": method,
        "metrics_units": "fractions in metrics; percentages in metrics_percent",
        "metrics": metrics, "metrics_percent": percentages,
        "model": summary,
        "evaluation_protocol": protocol,
        "training_protocol": training_protocol,
        "dataset_metadata": dataset_metadata,
        "selection_rule": (training_protocol or protocol).get("selection_rule", "not recorded"),
        "source_comparison": compare_source(task, split, metrics, summary, source_baseline_path),
        "contains_weights": False,
    }
    if "confusion_matrix" in metrics:
        report["confusion_matrix_axes"] = {"rows": "ground_truth", "columns": "prediction",
                                           "class_order": list(metrics["iou_by_class"])}
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
    return report
