"""Audit DGLE ISPRS manifests and sampled patch pixels against source archives."""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dgle_remote_sensing import (  # noqa: E402
    PD_EVAL,
    PD_TRAIN,
    VH_TRAIN,
    color_mask_to_ids,
    crop_positions,
    isprs_index,
)


ROOT = Path(__file__).resolve().parents[1] / "data" / "remote_sensing"
OUT = ROOT / "dgle"
SPECS = (
    ("vh_irrg_train", "Vaihingen", "ISPRS_semantic_labeling_Vaihingen.zip",
     "ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip",
     tuple(map(str, VH_TRAIN))),
    ("pd_rgb_train", "Potsdam", "2_Ortho_RGB.zip", None, PD_TRAIN),
    ("pd_rgb_eval", "Potsdam", "2_Ortho_RGB.zip", "5_Labels_all_noBoundary.zip", PD_EVAL),
)


def member_image(archive: zipfile.ZipFile, member: str) -> Image.Image:
    with archive.open(member) as stream:
        payload = stream.read()
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def main() -> None:
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    label_variant = metadata["isprs_vaihingen_labels"]
    if label_variant not in ("eroded", "complete"):
        raise AssertionError(f"Unknown Vaihingen label variant: {label_variant}")
    vh_chosen = ("ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip"
                 if label_variant == "eroded" else
                 "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip")
    vh_alternate = ("ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip"
                    if label_variant == "eroded" else
                    "ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip")
    pd_variant = metadata.get("isprs_potsdam_eval_labels", "no_boundary")
    if pd_variant not in ("no_boundary", "complete"):
        raise AssertionError(f"Unknown Potsdam eval label variant: {pd_variant}")
    pd_chosen = "5_Labels_all.zip" if pd_variant == "complete" else "5_Labels_all_noBoundary.zip"
    pd_alternate = "5_Labels_all_noBoundary.zip" if pd_variant == "complete" else "5_Labels_all.zip"
    specs = ((SPECS[0][0], SPECS[0][1], SPECS[0][2], vh_chosen, SPECS[0][4]),
             SPECS[1],
             (SPECS[2][0], SPECS[2][1], SPECS[2][2], pd_chosen, SPECS[2][4]))
    for split, dataset, image_name, label_name, tile_ids in specs:
        archive_dir = ROOT / "isprs" / "raw" / dataset
        with (OUT / "manifests" / f"isprs_{split}.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        by_tile: dict[str, dict[tuple[int, int], dict[str, str]]] = defaultdict(dict)
        for row in rows:
            stem = Path(row["image"]).stem
            parts = stem.split("_")
            tile_id = parts[0] if dataset == "Vaihingen" else "_".join(parts[:2])
            x, y = map(int, parts[1:] if dataset == "Vaihingen" else parts[2:])
            if (x, y) in by_tile[tile_id]:
                raise AssertionError(f"Duplicate coordinate {split}/{tile_id}/{x}/{y}")
            by_tile[tile_id][x, y] = row
        if set(by_tile) != set(tile_ids):
            raise AssertionError(f"Wrong tiles in {split}: {sorted(by_tile)}")
        exact_images = 0
        exact_labels = 0
        counts = Counter()
        with zipfile.ZipFile(archive_dir / image_name) as image_zip:
            images = isprs_index(image_zip, dataset, "image")
            label_zip = zipfile.ZipFile(archive_dir / label_name) if label_name else None
            try:
                labels = isprs_index(label_zip, dataset, "label") if label_zip else {}
                for tile in tile_ids:
                    raw = member_image(image_zip, images[tile])
                    positions = {(x, y) for y in crop_positions(raw.height) for x in crop_positions(raw.width)}
                    if positions != set(by_tile[tile]):
                        raise AssertionError(f"Wrong patch coordinates in {split}/{tile}")
                    label = member_image(label_zip, labels[tile]) if label_zip else None
                    if label and label.size != raw.size:
                        raise AssertionError(f"Image/label dimensions differ in {split}/{tile}")
                    samples = sorted(positions)
                    samples = [samples[i] for i in sorted({0, len(samples) // 2, len(samples) - 1})]
                    for x, y in samples:
                        row = by_tile[tile][x, y]
                        expected = np.asarray(raw.crop((x, y, x + 512, y + 512)))
                        with Image.open(OUT / row["image"]) as actual_image:
                            if actual_image.mode != "RGB" or not np.array_equal(np.asarray(actual_image), expected):
                                raise AssertionError(f"Image pixel mismatch: {row['image']}")
                        exact_images += 1
                        if label:
                            expected_ids = np.asarray(color_mask_to_ids(label.crop((x, y, x + 512, y + 512))))
                            with Image.open(OUT / row["label"]) as actual_label:
                                if actual_label.mode != "L" or not np.array_equal(np.asarray(actual_label), expected_ids):
                                    raise AssertionError(f"Label pixel mismatch: {row['label']}")
                            exact_labels += 1
                            values, frequencies = np.unique(expected_ids, return_counts=True)
                            counts.update(dict(zip(values.tolist(), frequencies.tolist())))
                    print(f"{split}: tile {tile}, raw {raw.width}x{raw.height}, patches {len(positions)}")
            finally:
                if label_zip:
                    label_zip.close()
        print(f"{split}: {len(rows)} manifest rows; {exact_images} image patches and "
              f"{exact_labels} label patches pixel-exact; sampled label IDs {dict(sorted(counts.items()))}")
        label_dir = OUT / "isprs" / split / "labels"
        if label_name is None and label_dir.exists():
            raise AssertionError(f"Unlabeled target split unexpectedly has labels directory: {label_dir}")

    variants = (
        ("Vaihingen source", "Vaihingen", vh_chosen, vh_alternate,
         tuple(map(str, VH_TRAIN))),
        ("Potsdam evaluation", "Potsdam", pd_chosen, pd_alternate, PD_EVAL),
    )
    for name, dataset, chosen_zip, alternate_zip, tiles in variants:
        base = ROOT / "isprs" / "raw" / dataset
        changed = chosen_black = alternate_black = pixels = 0
        with zipfile.ZipFile(base / chosen_zip) as chosen, zipfile.ZipFile(base / alternate_zip) as alternate:
            chosen_index = isprs_index(chosen, dataset, "label")
            alternate_index = isprs_index(alternate, dataset, "label")
            for tile in tiles:
                first = np.asarray(member_image(chosen, chosen_index[tile]))
                second = np.asarray(member_image(alternate, alternate_index[tile]))
                if first.shape != second.shape:
                    raise AssertionError(f"Label variant sizes differ for {name}/{tile}")
                changed += int(np.count_nonzero(np.any(first != second, axis=2)))
                chosen_black += int(np.count_nonzero(np.all(first == 0, axis=2)))
                alternate_black += int(np.count_nonzero(np.all(second == 0, axis=2)))
                pixels += first.shape[0] * first.shape[1]
        print(f"{name}: chosen versus alternate labels differ at {changed}/{pixels} "
              f"pixels ({changed / pixels:.2%}); black pixels chosen={chosen_black / pixels:.2%}, "
              f"alternate={alternate_black / pixels:.2%}")


if __name__ == "__main__":
    main()
