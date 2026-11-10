"""Cache physical/generic target views from original tiles; never open train GT."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def view_id(ratio):
    return "ratio_" + format(ratio, ".6g").replace(".", "p")


def make_view(tile, x, y, size, ratio):
    crop = round(size * ratio)
    left, top = math.floor(x + (size - crop) / 2), math.floor(y + (size - crop) / 2)
    image = tile.crop((left, top, left + crop, top + crop)).resize(
        (size, size), Image.Resampling.LANCZOS)
    observed = Image.new("L", (crop, crop))
    observed.paste(255, (max(0, -left), max(0, -top),
                        min(crop, tile.width - left), min(crop, tile.height - top)))
    observed = observed.resize((size, size), Image.Resampling.NEAREST)
    geometry = dict(native_size=size, native_x=x, native_y=y,
                    view_size=crop, view_x=left, view_y=top)
    axis = np.arange(size)
    vx = (axis + x - left + .5) * size / crop - .5
    vy = (axis + y - top + .5) * size / crop - .5
    xx, yy = np.meshgrid(vx, vy)
    valid = (xx >= 0) & (xx <= size - 1) & (yy >= 0) & (yy <= size - 1)
    support = np.asarray(observed) > 0
    for offset_x, offset_y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ix = np.clip(np.floor(xx).astype(int) + offset_x, 0, size - 1)
        iy = np.clip(np.floor(yy).astype(int) + offset_y, 0, size - 1)
        valid &= support[iy, ix]
    return image, observed, geometry, valid


def choose_diagnostic(rows, count):
    groups = {}
    for row in rows:
        name = Path(row["image"]).name
        tile = "_".join(Path(name).stem.split("_")[:2])
        groups.setdefault(tile, []).append(name)
    selected = []
    # Stable, image-only sampling across parent tiles.
    ordered = {tile: sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
               for tile, names in groups.items()}
    for index in range(max(map(len, ordered.values()))):
        for tile in sorted(ordered):
            if index < len(ordered[tile]):
                selected.append(ordered[tile][index])
                if len(selected) >= count:
                    return set(selected)
    return set(selected)


def prepare(base, archive_path, output, bands, ratios, diagnostic_limit, train_limit=0, eval_limit=0):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "gsd_views"
    cache.mkdir(exist_ok=True)
    args_card = {"bands": bands, "ratios": ratios, "diagnostic_limit": diagnostic_limit,
                 "train_limit": train_limit, "eval_limit": eval_limit,
                 "archive_sha256": digest(archive_path)}
    existing = cache / "view_manifest.json"
    if existing.is_file():
        state = json.loads(existing.read_text())
        if state["preparation"] != args_card:
            raise ValueError("Existing cache has another preparation; choose a new output")
        print(f"Already prepared {output}", flush=True)
        return
    manifests = output / "manifests"
    manifests.mkdir(exist_ok=True)
    for name in ("source_train", "source_val"):
        shutil.copy2(base / "manifests" / f"{name}.csv", manifests / f"{name}.csv")
    rows_by_tile = {}
    diagnostics = set()
    counts = {}
    for split in ("target_train", "target_eval"):
        with (base / "manifests" / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if split == "target_train":
            if any(row["label"] for row in rows):
                raise ValueError("Target train manifest must be image-only")
            if train_limit:
                names = choose_diagnostic(rows, train_limit)
                rows = [row for row in rows if Path(row["image"]).name in names]
        else:
            diagnostics = choose_diagnostic(rows, diagnostic_limit)
            if eval_limit:
                names = choose_diagnostic(rows, eval_limit)
                rows = [row for row in rows if Path(row["image"]).name in names]
                diagnostics &= names
        counts[split] = len(rows)
        with (manifests / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image", "label"])
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            stem = Path(row["image"]).stem
            parts = stem.split("_")
            if len(parts) != 4 or not all(x.isdigit() for x in parts):
                raise ValueError(f"Unexpected patch coordinates: {stem}")
            tile = "_".join(parts[:2])
            rows_by_tile.setdefault(tile, []).append((split, row, int(parts[2]), int(parts[3])))
    records = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for name in archive.namelist():
            match = re.search(r"top_potsdam_(\d+_\d+)", Path(name).name)
            if match and name.lower().endswith(".tif"):
                members[match[1]] = name
        for tile_name, rows in sorted(rows_by_tile.items()):
            if tile_name not in members:
                raise FileNotFoundError(f"Missing parent tile {tile_name}")
            with archive.open(members[tile_name]) as stream, Image.open(stream) as loaded:
                tile = loaded.convert("RGB")
                tile.load()
            for split, row, x, y in rows:
                native = output / row["image"]
                native.parent.mkdir(parents=True, exist_ok=True)
                tile.crop((x, y, x + 512, y + 512)).save(native, compress_level=1)
                if split == "target_eval":
                    destination = output / row["label"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(base / row["label"], destination)
                name = native.name
                if split == "target_eval" and name not in diagnostics:
                    continue
                record = {"split": split, "image": row["image"], "label": row["label"],
                          "tile": tile_name, "gsd_m": .05, "views": {}}
                common = np.ones((512, 512), dtype=bool)
                for ratio in ratios:
                    key = view_id(ratio)
                    image, mask, geometry, valid = make_view(tile, x, y, 512, ratio)
                    image_path = Path(key) / name
                    valid_path = Path(key + "_valid") / name
                    for relative in (image_path, valid_path):
                        (cache / relative).parent.mkdir(parents=True, exist_ok=True)
                    image.save(cache / image_path, compress_level=1)
                    mask.save(cache / valid_path)
                    record["views"][key] = {"image": image_path.as_posix(),
                        "valid": valid_path.as_posix(), "geometry": geometry,
                        "actual_gsd_m": .05 * geometry["view_size"] / 512}
                    common &= valid
                relative = Path("common_native_valid") / name
                (cache / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(common.astype(np.uint8) * 255).save(cache / relative)
                record["common_native_valid"] = relative.as_posix()
                records[name] = record
            print(f"Prepared {tile_name}: {len(rows)} native patches", flush=True)
            del tile
    card = {"format": 1, "bands": bands, "target_gsd_m": .05,
            "preparation": args_card, "records": records,
            "diagnostic_names": sorted(diagnostics), "train_gt_opened": 0,
            "source_manifest_sha256": digest(manifests / "source_train.csv")}
    existing.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    metadata = json.loads((base / "metadata.json").read_text())
    metadata.update(gsd_study_variant=bands, gsd_view_manifest_sha256=digest(existing),
                    parent_tile_views=True, target_train_labels=False)
    metadata["counts"].update(counts)
    metadata["manifest_sha256"] = {p.stem: digest(p) for p in manifests.glob("*.csv")}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} view records; train GT opened=0", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bands", choices=("IRRG", "RGB"), required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.8, 2.])
    parser.add_argument("--diagnostic-limit", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()
    if args.diagnostic_limit < 1 or min(args.train_limit, args.eval_limit) < 0 or any(not .5 <= x <= 2 for x in args.ratios):
        parser.error("Invalid sampling limit or view ratio")
    prepare(args.base_data, args.archive, args.output, args.bands,
            args.ratios, args.diagnostic_limit, args.train_limit, args.eval_limit)


if __name__ == "__main__":
    main()
