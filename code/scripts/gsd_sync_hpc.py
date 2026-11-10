"""Install the isolated study snapshot and export only required inference tensors."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if root.name != "gsd_sfuda_20261012" or root.parent != Path.home() / "Tu":
        raise ValueError("Unexpected destination")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        inventory = json.loads(archive.read("code_snapshot_manifest.json"))
        for item in inventory:
            destination = (root / item["path"]).resolve()
            if root not in destination.parents:
                raise ValueError("Snapshot path escapes study root")
            content = archive.read(item["path"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("Snapshot hash mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    old = Path.home() / "Tu/hf_remote_sensing_bundle"
    data = root / "data/remote_sensing"
    data.mkdir(parents=True, exist_ok=True)
    link = data / "siamseg_sfuda_b2_v1"
    if not link.exists():
        link.symlink_to(old / "data/remote_sensing/siamseg_sfuda_b2_v1", target_is_directory=True)
    if not (root / "pretrain").exists():
        (root / "pretrain").symlink_to(old / "pretrain", target_is_directory=True)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    source = old / "results_remote_sensing/siamseg_sfuda_b2_v1/vh2pd/source/seed1/last.pth"
    anchor = checkpoints / "source_resume_08500.pth"
    if not anchor.exists():
        shutil.copy2(source, anchor)
    print(json.dumps({"root": str(root), "code_files": len(inventory),
                      "source_resume_sha256": digest(anchor)}, indent=2))


if __name__ == "__main__":
    main()
