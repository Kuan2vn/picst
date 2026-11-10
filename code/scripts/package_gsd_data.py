"""Archive only explicitly selected prepared datasets, with a transfer hash."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = args.dataset.resolve()
    if root / "data/remote_sensing" != dataset.parent or not (dataset / "gsd_views/view_manifest.json").is_file():
        raise ValueError("Expected a complete prepared GSD dataset in the workspace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in dataset.rglob("*") if p.is_file())
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    value = hashlib.sha256()
    with args.output.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    card = {"archive": args.output.name, "bytes": args.output.stat().st_size,
            "sha256": value.hexdigest(), "files": len(paths), "dataset": dataset.name}
    args.output.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card), flush=True)


if __name__ == "__main__":
    main()
