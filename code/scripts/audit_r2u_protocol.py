"""Audit existing R2U run provenance without torch, CUDA or training.

Checks manifests/files and recorded hashes. It does not inspect checkpoint
tensors or reproduce evaluation; those are separate gates.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SPLITS = {"loveda_rural_train": (1366, True),
          "loveda_urban_train": (1156, False), "loveda_urban_val": (677, True)}
CLASSES = {"background", "building", "road", "water", "barren", "forest", "agriculture"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(repo: Path, data: Path, source: Path) -> dict:
    repo, data, source = repo.resolve(), data.resolve(), source.resolve()
    metadata = read_json(data / "metadata.json")
    require(metadata.get("ignore_id") == 255, "Expected ignore ID 255")
    require(metadata.get("target_training_labels_excluded") is True, "Target labels not excluded")
    splits, image_sets, manifest_hashes = {}, {}, {}
    for name, (count, labeled) in SPLITS.items():
        manifest = data / "manifests" / (name + ".csv")
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == count, f"Incorrect count for {name}: {len(rows)}")
        require(metadata["counts"]["loveda"][name.removeprefix("loveda_")] == count,
                "Metadata count mismatch: " + name)
        images = set()
        for row in rows:
            require(set(row) == {"image", "label"}, "Malformed manifest")
            require(bool(row["label"]) == labeled, "Incorrect label visibility: " + name)
            for column in ("image", "label"):
                if row[column]:
                    path = (data / row[column]).resolve()
                    require(data in path.parents and path.is_file(), "Missing/unsafe " + str(path))
            image = (data / row["image"]).resolve()
            require(image not in images, "Duplicate image in " + name)
            images.add(image)
        image_sets[name] = images
        manifest_hashes[name] = digest(manifest)
        splits[name] = {"images": count, "has_label_paths": labeled,
                        "manifest_sha256": manifest_hashes[name]}
    require(not (data / "loveda" / "urban_train" / "labels").exists(),
            "Target train label directory is present")
    names = list(SPLITS)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            require(not image_sets[name] & image_sets[other], "Split image paths overlap")

    training = read_json(source / "protocol_source.json")
    evaluation = read_json(source / "source_eval" / "protocol_eval.json")
    baseline = read_json(source / "source_eval" / "metrics_eval.json")
    require(training["task"] == "r2u" and training["stage"] == "source" and
            training["split"] == "loveda_rural_train", "Incorrect source split/stage")
    require(evaluation["task"] == "r2u" and evaluation["stage"] == "eval" and
            evaluation["split"] == "loveda_urban_val", "Incorrect evaluation split/stage")
    require(baseline["task"] == "r2u" and baseline["split"] == "loveda_urban_val",
            "Incorrect metric split")
    checkpoint = source / "source_last.pth"
    checkpoint_hash = digest(checkpoint)
    require(baseline["source_checkpoint_sha256"] == checkpoint_hash,
            "Evaluation metric does not match current source checkpoint")
    require(evaluation["files_sha256"]["source_checkpoint"] == checkpoint_hash,
            "Evaluation protocol does not match current source checkpoint")
    for protocol, split in ((training, "loveda_rural_train"), (evaluation, "loveda_urban_val")):
        require(protocol["files_sha256"]["manifest"] == manifest_hashes[split],
                "Manifest changed since recorded run: " + split)
        require(protocol["files_sha256"]["dataset_metadata"] == digest(data / "metadata.json"),
                "Dataset metadata changed since recorded run")
    backbone_path = Path(training["args"]["backbone_weights"])
    if not backbone_path.is_absolute():
        backbone_path = repo / backbone_path
    require(digest(backbone_path) == training["files_sha256"]["initial_backbone"],
            "Initial backbone file hash mismatch")
    metrics = baseline["metrics"]
    require(len(metrics["iou"]) == 7 and set(metrics["iou_by_class"]) == CLASSES,
            "Incorrect metric class mapping")
    require(0 < metrics["valid_pixels"] <= 677 * 1024 * 1024, "Invalid evaluation pixel count")
    require(abs(sum(metrics["iou"]) / 7 - metrics["miou"]) < 1e-9, "mIoU mismatch")

    code_history = {}
    for field, name in {"trainer": "train_SRMP_remote.py",
                        "transforms": "scripts/remote_sensing_transforms.py"}.items():
        current = digest(repo / name)
        recorded = training["files_sha256"].get(field)
        code_history[name] = {"recorded_training_sha256": recorded,
                              "current_sha256": current, "unchanged": recorded == current}
    adaptations = []
    for protocol_path in sorted(source.parent.glob("rel*/srmp/protocol_adapt.json")):
        protocol = read_json(protocol_path)
        require(protocol["task"] == "r2u" and protocol["split"] == "loveda_urban_train",
                "Incorrect adaptation split")
        require(protocol["files_sha256"]["source_checkpoint"] == checkpoint_hash,
                "Adaptation initialized from another source checkpoint")
        require(protocol["files_sha256"]["manifest"] == manifest_hashes["loveda_urban_train"],
                "Adaptation manifest changed")
        adaptations.append({"path": str(protocol_path), "method": "our_srmp",
                            "relation_weight": protocol["args"]["relation_weight"],
                            "source_checkpoint_matches": True,
                            "planned_steps": protocol["args"]["steps"]})
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "r2u", "status": "structural_and_provenance_checks_passed",
        "published_method_reproduction_verified": False, "contains_weights": False,
        "splits": splits, "source_checkpoint_sha256": checkpoint_hash,
        "source_training_args": training["args"], "source_versions": training["versions"],
        "source_metrics": metrics, "source_miou_percent": metrics["miou"] * 100,
        "code_history": code_history, "existing_adaptations": adaptations,
        "remaining_gates": [
            "Inspect checkpoint metadata/tensor compatibility in the training Python environment",
            "Verify all mask IDs and raw-to-prepared label mapping; this audit reads no pixel masks",
            "No image-content duplicate check; only file-path separation was checked",
            "Re-evaluate with the recorded inference policy and verify final training iteration",
            "Implement/validate a published SFUDA baseline; existing adapt runs are our SRMP",
            "DGLE does not expose all source-training/remote-baseline configurations",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = args.data_root or args.repo / "data/remote_sensing/dgle"
    source = args.source_root or args.repo / (
        "results_remote_sensing/r2u/rs_standard/crop512_batch4_accum1_source20000/seed1/source")
    report = audit(args.repo, data, source)
    output = args.output or source / "audit_r2u_protocol.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "source_miou_percent": report["source_miou_percent"],
                      "published_method_reproduction_verified": False}, indent=2))


if __name__ == "__main__":
    main()
