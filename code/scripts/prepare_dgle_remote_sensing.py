"""Prepare the two DGLE remote-sensing benchmarks from verified source ZIPs.

Output keeps target training labels absent, as required by source-free adaptation.
Native mask values are converted to contiguous train IDs; 255 means ignore.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "data" / "remote_sensing"
VH_TRAIN = (1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34, 37)
# Held-out source validation, added by us; not specified as validation in DGLE.
VH_VAL = (5, 15, 21, 30)
PD_TRAIN = ("6_10", "7_10", "2_12", "3_11", "2_10", "7_8", "5_10", "3_12", "5_12", "7_11", "7_9", "6_9", "7_7", "4_12", "6_8", "6_12", "6_7", "4_11")
PD_EVAL = ("2_11", "3_10", "4_10", "5_11", "6_11", "7_12")
ISPRS_COLORS = {
    (255, 255, 255): 0,  # Impervious surface
    (0, 0, 255): 1,      # Building
    (0, 255, 255): 2,    # Low vegetation
    (0, 255, 0): 3,      # Tree
    (255, 255, 0): 4,    # Car
    (255, 0, 0): 5,      # Clutter
}
LOVE_LUT = np.full(256, 255, dtype=np.uint8)
LOVE_LUT[1:8] = np.arange(7, dtype=np.uint8)


def write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "label"))
        writer.writerows(rows)


def loveda_members(archive: zipfile.ZipFile, split: str, domain: str, kind: str) -> dict[str, str]:
    prefix = (split, domain, kind)
    found = {}
    for name in archive.namelist():
        parts = PurePosixPath(name).parts
        if len(parts) == 4 and parts[:3] == prefix and parts[3].lower().endswith(".png"):
            found[Path(parts[3]).stem] = name
    return found


def prepare_loveda(root: Path, output: Path, only_split: str | None = None) -> dict[str, int]:
    raw = root / "LoveDA" / "raw"
    specs = (
        ("Train", "Rural", "rural_train", True, 1366),
        ("Train", "Urban", "urban_train", False, 1156),
        ("Val", "Urban", "urban_val", True, 677),
    )
    counts = {}
    for split, domain, output_name, include_labels, expected in specs:
        if only_split and output_name != only_split:
            continue
        archive_path = raw / f"{split}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing {archive_path}; run download_dgle_remote_sensing.py")
        with zipfile.ZipFile(archive_path) as archive:
            images = loveda_members(archive, split, domain, "images_png")
            masks = loveda_members(archive, split, domain, "masks_png") if include_labels else {}
            if len(images) != expected:
                raise RuntimeError(f"{split}/{domain}: found {len(images)} images, expected {expected}")
            if include_labels and set(images) != set(masks):
                raise RuntimeError(f"{split}/{domain}: image/mask identifiers differ")
            rows = []
            for number, stem in enumerate(sorted(images, key=lambda x: int(x) if x.isdigit() else x), 1):
                image_rel = f"loveda/{output_name}/images/{stem}.png"
                image_path = output / image_rel
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not image_path.exists():
                    with archive.open(images[stem]) as source, image_path.open("wb") as dest:
                        shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
                label_rel = ""
                if include_labels:
                    label_rel = f"loveda/{output_name}/labels/{stem}.png"
                    label_path = output / label_rel
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    if not label_path.exists():
                        with archive.open(masks[stem]) as source, Image.open(source) as native:
                            arr = np.asarray(native)
                            if arr.ndim != 2 or arr.max() > 7:
                                raise RuntimeError(f"Unexpected LoveDA mask encoding: {masks[stem]}")
                            Image.fromarray(LOVE_LUT[arr], mode="L").save(label_path)
                rows.append((image_rel, label_rel))
                if number % 200 == 0 or number == len(images):
                    print(f"LoveDA {output_name}: {number}/{len(images)}", flush=True)
            write_manifest(output / "manifests" / f"loveda_{output_name}.csv", rows)
            counts[output_name] = len(rows)
    return counts


def isprs_index(archive: zipfile.ZipFile, dataset: str, role: str) -> dict[str, str]:
    result = {}
    for name in archive.namelist():
        if dataset == "Vaihingen" and role == "image" and PurePosixPath(name).parts[0].lower() != "top":
            continue
        basename = PurePosixPath(name).name
        if not basename.lower().endswith(".tif"):
            continue
        match = re.search(r"top_potsdam_(\d+_\d+)", basename, re.I) if dataset == "Potsdam" else re.search(r"area(\d+)", basename, re.I)
        if match:
            key = match.group(1)
            if key in result:
                raise RuntimeError(f"Duplicate ISPRS tile {key} in {archive.filename}")
            result[key] = name
    return result


def crop_positions(length: int, crop: int = 512) -> list[int]:
    if length < crop:
        raise RuntimeError(f"Image width/height {length} smaller than {crop}")
    return sorted({min(position, length - crop) for position in range(0, length, crop)})


def color_mask_to_ids(label: Image.Image) -> Image.Image:
    rgb = np.asarray(label.convert("RGB"))
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, train_id in ISPRS_COLORS.items():
        ids[np.all(rgb == color, axis=2)] = train_id
    unknown = (ids == 255) & np.any(rgb != 0, axis=2)
    if unknown.any():
        colors = np.unique(rgb[unknown].reshape(-1, 3), axis=0)[:8]
        raise RuntimeError(f"Unexpected ISPRS label colors: {colors.tolist()}")
    return Image.fromarray(ids, mode="L")


def prepare_isprs(root: Path, output: Path, eroded: bool, only_split: str | None = None,
                  pd_eval_complete: bool = False) -> dict[str, int]:
    raw = root / "isprs" / "raw"
    vh_label_file = "ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip" if eroded else "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip"
    pd_eval_label_file = "5_Labels_all.zip" if pd_eval_complete else "5_Labels_all_noBoundary.zip"
    specs = (
        ("Vaihingen", "ISPRS_semantic_labeling_Vaihingen.zip", vh_label_file, "vh_irrg_train", [str(x) for x in VH_TRAIN], True),
        ("Potsdam", "2_Ortho_RGB.zip", "5_Labels_all_noBoundary.zip", "pd_rgb_train", list(PD_TRAIN), False),
        ("Potsdam", "2_Ortho_RGB.zip", pd_eval_label_file, "pd_rgb_eval", list(PD_EVAL), True),
    )
    if only_split == "vh_irrg_val":
        specs = (("Vaihingen", "ISPRS_semantic_labeling_Vaihingen.zip", vh_label_file,
                  "vh_irrg_val", list(map(str, VH_VAL)), True),)
    counts = {}
    for dataset, image_zip_name, label_zip_name, output_name, identifiers, include_labels in specs:
        if only_split and output_name != only_split:
            continue
        dataset_dir = raw / dataset
        image_zip_path = dataset_dir / image_zip_name
        label_zip_path = dataset_dir / label_zip_name
        if not image_zip_path.is_file() or (include_labels and not label_zip_path.is_file()):
            raise FileNotFoundError(f"Missing ISPRS archives in {dataset_dir}; run download_dgle_isprs.py")
        with zipfile.ZipFile(image_zip_path) as images_zip:
            images = isprs_index(images_zip, dataset, "image")
            labels_zip = zipfile.ZipFile(label_zip_path) if include_labels else None
            try:
                labels = isprs_index(labels_zip, dataset, "label") if labels_zip else {}
                missing = set(identifiers) - set(images)
                if include_labels:
                    missing |= set(identifiers) - set(labels)
                if missing:
                    raise RuntimeError(f"{dataset}/{output_name}: missing tiles {sorted(missing)}")
                rows = []
                for tile_no, tile_id in enumerate(identifiers, 1):
                    with images_zip.open(images[tile_id]) as source:
                        image_bytes = source.read()
                    with Image.open(io.BytesIO(image_bytes)) as native_image:
                        rgb = native_image.convert("RGB")
                    label_image = None
                    if include_labels and labels_zip:
                        with labels_zip.open(labels[tile_id]) as source:
                            label_bytes = source.read()
                        with Image.open(io.BytesIO(label_bytes)) as native_label:
                            if native_label.size != rgb.size:
                                raise RuntimeError(f"Image/label size mismatch for {dataset} {tile_id}")
                            label_image = native_label.convert("RGB")
                    for y in crop_positions(rgb.height):
                        for x in crop_positions(rgb.width):
                            stem = f"{tile_id}_{x}_{y}"
                            image_rel = f"isprs/{output_name}/images/{stem}.png"
                            image_path = output / image_rel
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            if not image_path.exists():
                                rgb.crop((x, y, x + 512, y + 512)).save(image_path)
                            label_rel = ""
                            if label_image is not None:
                                label_rel = f"isprs/{output_name}/labels/{stem}.png"
                                label_path = output / label_rel
                                label_path.parent.mkdir(parents=True, exist_ok=True)
                                # Refresh labels when switching Vaihingen variants; target
                                # training has no label archive or output mask.
                                color_mask_to_ids(label_image.crop((x, y, x + 512, y + 512))).save(label_path)
                            rows.append((image_rel, label_rel))
                    print(f"ISPRS {output_name}: tile {tile_no}/{len(identifiers)}, {len(rows)} patches", flush=True)
                write_manifest(output / "manifests" / f"isprs_{output_name}.csv", rows)
                counts[output_name] = len(rows)
            finally:
                if labels_zip:
                    labels_zip.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dataset", choices=("loveda", "isprs", "all"), default="all")
    parser.add_argument("--vaihingen-labels", choices=("eroded", "complete"), default="eroded")
    parser.add_argument("--potsdam-eval-labels", choices=("no_boundary", "complete"),
                        default="no_boundary")
    parser.add_argument("--isprs-only", choices=("vh_irrg_train", "vh_irrg_val", "pd_rgb_train", "pd_rgb_eval"))
    parser.add_argument("--loveda-only", choices=("rural_train", "urban_train", "urban_val"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "dgle"
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if args.isprs_only == "vh_irrg_val":
        if args.dataset != "isprs" or args.vaihingen_labels != "eroded":
            parser.error("Source validation requires --dataset isprs and eroded labels")
        counts = prepare_isprs(root, output, True, "vh_irrg_val")
        manifest = output / "manifests/isprs_vh_irrg_val.csv"
        info = {"split": "isprs_vh_irrg_val", "tile_ids": list(VH_VAL),
                "train_tile_ids": list(VH_TRAIN), "counts": counts,
                "labels": "eroded", "bands": "IRRG", "patch_size": 512,
                "purpose": "source-domain validation; extension to DGLE, never used for gradient updates",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        (output / "source_val_metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(info, indent=2))
        return  # Preserve all original manifests and metadata, including target masks.
    counts = {}
    if args.dataset in ("loveda", "all"):
        counts["loveda"] = prepare_loveda(root, output, args.loveda_only)
    if args.dataset in ("isprs", "all"):
        counts["isprs"] = prepare_isprs(
            root, output, args.vaihingen_labels == "eroded", args.isprs_only,
            args.potsdam_eval_labels == "complete")
    prior_counts = previous.get("counts", {})
    for dataset, split_counts in counts.items():
        prior_counts.setdefault(dataset, {}).update(split_counts)
    metadata = {
        "paper": "DGLE, IEEE JSTARS 2026, DOI 10.1109/JSTARS.2026.3693002",
        "counts": prior_counts,
        "isprs_patch_size": 512,
        "isprs_vaihingen_labels": (
            args.vaihingen_labels
            if args.dataset in ("isprs", "all") and args.isprs_only in (None, "vh_irrg_train")
            else previous.get("isprs_vaihingen_labels")
        ),
        "isprs_potsdam_eval_labels": (
            args.potsdam_eval_labels
            if args.dataset in ("isprs", "all") and args.isprs_only in (None, "pd_rgb_eval")
            else previous.get("isprs_potsdam_eval_labels", "no_boundary")
        ),
        "target_training_labels_excluded": True,
        "ignore_id": 255,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
