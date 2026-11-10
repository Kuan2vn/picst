"""Verify and extract prepared splits, optionally including VH source validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or len(manifest.get("archives", [])) not in (6, 7):
        raise ValueError("Unexpected bundle manifest")
    for item in manifest["artifacts"]:
        path = (ROOT / item["path"]).resolve()
        if os.path.commonpath((str(ROOT), str(path))) != str(ROOT):
            raise ValueError(f"Artifact path escapes bundle: {item['path']}")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Missing/corrupted artifact: {item['path']}")
        print(f"Verified {item['path']}", flush=True)
    for item in manifest["archives"]:
        archive_path = ROOT / item["path"]
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != item["files"]:
                raise ValueError(f"Archive count mismatch: {item['path']}")
            for name in names:
                destination = (ROOT / name).resolve()
                if os.path.commonpath((str(ROOT), str(destination))) != str(ROOT):
                    raise ValueError(f"Unsafe archive path: {name}")
            print(f"Extracting {item['path']}: {len(names)} files", flush=True)
            archive.extractall(ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_dgle_remote_sensing.py")],
                   cwd=ROOT, check=True)
    print("Bundle restored and all six DGLE splits verified.")


if __name__ == "__main__":
    main()
