"""Validate the prepared DGLE remote-sensing datasets and source-free splits."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "data" / "remote_sensing" / "dgle"
EXPECTED = {
    "isprs_vh_irrg_train": (256, True, (512, 512), set(range(6)) | {255}),
    "isprs_pd_rgb_train": (2592, False, (512, 512), None),
    "isprs_pd_rgb_eval": (864, True, (512, 512), set(range(6)) | {255}),
    "loveda_rural_train": (1366, True, (1024, 1024), set(range(7)) | {255}),
    "loveda_urban_train": (1156, False, (1024, 1024), None),
    "loveda_urban_val": (677, True, (1024, 1024), set(range(7)) | {255}),
}


def safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if os.path.commonpath((str(root), str(path))) != str(root):
        raise RuntimeError(f"Manifest path escapes dataset root: {value}")
    return path


def verify(root: Path, task: str | None = None) -> None:
    if task not in (None, "r2u", "vh2pd"):
        raise ValueError(f"Unknown task: {task}")
    expected = {name: spec for name, spec in EXPECTED.items()
                if task is None or name.startswith("loveda_" if task == "r2u" else "isprs_")}
    root = root.resolve()
    rows_by_split = {}
    for split, (expected_count, labeled, size, valid_ids) in expected.items():
        manifest = root / "manifests" / f"{split}.csv"
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected_count:
            raise RuntimeError(f"{split}: expected {expected_count} rows, got {len(rows)}")
        images = set()
        for row in rows:
            image = safe_path(root, row["image"])
            if not image.is_file() or row["image"] in images:
                raise RuntimeError(f"{split}: missing or duplicate image {row['image']}")
            images.add(row["image"])
            if bool(row["label"]) != labeled:
                raise RuntimeError(f"{split}: incorrect label field for {row['image']}")
            if labeled and not safe_path(root, row["label"]).is_file():
                raise RuntimeError(f"{split}: missing label {row['label']}")
        for index in sorted({0, len(rows) // 2, len(rows) - 1}):
            row = rows[index]
            with Image.open(safe_path(root, row["image"])) as image:
                image.load()
                if image.size != size or image.mode != "RGB":
                    raise RuntimeError(f"{split}: unexpected image format {row['image']}: {image.size}/{image.mode}")
            if labeled:
                with Image.open(safe_path(root, row["label"])) as label:
                    label.load()
                    ids = set(np.unique(np.asarray(label)).tolist())
                    if label.size != size or not ids <= valid_ids:
                        raise RuntimeError(f"{split}: unexpected label {row['label']}: {label.size}/{sorted(ids)}")
        rows_by_split[split] = rows
        if not labeled:
            label_dir = root / ("isprs" if split.startswith("isprs_") else "loveda")
            label_dir = label_dir / split.split("_", 1)[1] / "labels"
            if label_dir.exists():
                raise RuntimeError(f"{split}: target train label directory must be absent: {label_dir}")
        print(f"{split}: {len(rows)} OK")

    def pd_tile(row):
        return "_".join(Path(row["image"]).stem.split("_")[:2])

    if "isprs_pd_rgb_train" in rows_by_split:
        train_tiles = {pd_tile(row) for row in rows_by_split["isprs_pd_rgb_train"]}
        eval_tiles = {pd_tile(row) for row in rows_by_split["isprs_pd_rgb_eval"]}
        if train_tiles & eval_tiles or len(train_tiles) != 18 or len(eval_tiles) != 6:
            raise RuntimeError("Potsdam train/eval tile split is invalid")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    for split, (count, *_rest) in expected.items():
        dataset, name = split.split("_", 1)
        if metadata["counts"][dataset][name] != count:
            raise RuntimeError(f"metadata.json count mismatch: {split}")
    print(f"DGLE {task or 'both tasks'} splits verified.")


if __name__ == "__main__":
    verify(ROOT)
