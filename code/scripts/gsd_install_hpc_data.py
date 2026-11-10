"""Verify and install the full IRRG archive in the explicitly named HPC study."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    args = parser.parse_args()
    root = (Path.home() / "Tu/gsd_sfuda_20261012").resolve()
    card = json.loads(args.card.read_text())
    digest = hashlib.sha256()
    with args.archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != card["sha256"]:
        raise ValueError("Full data archive SHA-256 mismatch")
    prefix = "data/remote_sensing/gsd_sfuda_b2_irrg_v1/"
    with zipfile.ZipFile(args.archive) as archive:
        for item in archive.infolist():
            destination = (root / item.filename).resolve()
            if (not item.filename.startswith(prefix) or root not in destination.parents or
                    ((item.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe archive path")
        archive.extractall(root)
    data = root / prefix
    manifest = json.loads((data / "gsd_views/view_manifest.json").read_text())
    for name, item in manifest["records"].items():
        required = [data / item["image"], data / "gsd_views" / item["common_native_valid"]]
        if item["label"]:
            if item["split"] != "target_eval":
                raise ValueError("Target adaptation GT must not be shipped")
            required.append(data / item["label"])
        for view in item["views"].values():
            required.extend([data / "gsd_views" / view["image"], data / "gsd_views" / view["valid"]])
        if not all(path.is_file() for path in required):
            raise ValueError(f"Missing cached files: {name}")
    record = {"archive_sha256": digest.hexdigest(), "view_records": len(manifest["records"]),
              "diagnostic_images": len(manifest["diagnostic_names"]), "verified": True}
    (root / "data_install_verified.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
