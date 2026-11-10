"""Assemble the authorized lightweight public package using an explicit allowlist."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "hf_gsd_sfuda_bundle"
    sources = {
        "gsd_code_20261012.zip": ROOT / "artifacts/gsd_code_20261012.zip",
        "gsd_kaggle_irrg_v1.zip": ROOT / "artifacts/gsd_kaggle_irrg_v1.zip",
        "checkpoints/source_08500.pth": ROOT / "artifacts/gsd_public_checkpoints/source_08500.pth",
        "checkpoints/source_smoke.pth": ROOT / "artifacts/gsd_public_checkpoints/source_smoke.pth",
        "checkpoints/CHECKPOINTS.json": ROOT / "artifacts/gsd_public_checkpoints/CHECKPOINTS.json",
        "gsd_sfuda_kaggle.ipynb": ROOT / "notebooks/gsd_sfuda_kaggle.ipynb",
        "README.md": ROOT / "GSD_HF_README.md",
    }
    items = []
    for name, path in sources.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        value = hashlib.sha256()
        with destination.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(block)
        items.append({"path": name, "sha256": value.hexdigest(), "bytes": destination.stat().st_size})
    manifest = {"format": 1, "purpose": "GSD-SFUDA development diagnostics and integration smoke",
                "files": items, "target_train_images": 16, "target_eval_images": 128,
                "target_eval_parent_tiles": 14, "target_train_labels": False,
                "source_training_images_included": False, "main_training_results_included": False,
                "credentials_included": False}
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"public_folder": str(output), "files": len(items) + 1,
                      "bytes": sum(item["bytes"] for item in items)}))


if __name__ == "__main__":
    main()
