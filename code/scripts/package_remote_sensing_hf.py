"""Build the self-contained DGLE remote-sensing HPC bundle for Hugging Face.

The six benchmark splits and held-out VH source validation are archived. Original downloads,
training results, unrelated datasets, and local credentials are never copied.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf_remote_sensing_bundle"
DATA = ROOT / "data" / "remote_sensing" / "dgle"
SPLITS = (
    ("isprs", "vh_irrg_train"),
    ("isprs", "vh_irrg_val"),
    ("isprs", "pd_rgb_train"),
    ("isprs", "pd_rgb_eval"),
    ("loveda", "rural_train"),
    ("loveda", "urban_train"),
    ("loveda", "urban_val"),
)
ROOT_FILES = (
    "run_source_vh2pd.sh", "SOURCE_VH2PD_RETRAIN.md", "tests/test_source_remote.py",
    "references/siamseg/LICENSE",
    "train_SND.py", "train_SND_remote.py", "run_snd_vh2pd.sh", "configs/snd_vh2pd.json", "REPRODUCE_SND_VH2PD.md", "tests/test_snd_remote.py",
    "train_SRMP_remote.py", "train_SRMP.py", "train_ReMask.py",
    "remote_sensing_srmp.slurm", "run_remote_sensing_hpc.sh",
    "reproduce_dgle_source_only.sh",
    "setup_dgle_remote_sensing_hpc.sh",
    "requirements.txt", "REPRODUCE_DGLE_SOURCE_ONLY.md", "REPRODUCE_DGLE_SFUDA.md",
    "REMOTE_SENSING_DGLE_SETUP.md", "REMOTE_SENSING_PIPELINE_AUDIT.md",
    "train_CROTS_remote.py", "run_crots_r2u.sh", "requirements_crots.txt",
    "configs/crots_r2u.json", "REPRODUCE_CROTS_R2U.md",
    "train_CROTS_whitebox.py", "run_crots_whitebox.sh", "CROTS_HPC_AUDIT.md",
    "configs/crots_whitebox_r2u.json", "configs/crots_whitebox_vh2pd.json",
    "references/crots/manifest.json", "references/crots/README.md",
    "references/crots/modules/smhm.py", "references/crots/modules/basic.py",
    "references/crots/generate_pseudo_labels.py", "references/crots/construct_hard_patch.py",
    "references/crots/utils/transformsgpu.py", "references/crots/models/deeplabv2.py",
)
SCRIPT_FILES = (
    "snd_remote.py", "probe_crots_failure.py", "diagnose_source_curve.py",
    "__init__.py", "remote_sensing_transforms.py", "remote_sensing_results.py",
    "download_dgle_remote_sensing.py", "download_dgle_isprs.py",
    "dgle_range_download.py",
    "verify_dgle_remote_sensing.py", "compare_dgle_source_only.py", "audit_r2u_protocol.py",
    "prepare_dgle_remote_sensing.py", "audit_dgle_isprs.py",
    "fetch_remote_backbone.py", "restore_hf_remote_bundle.py",
    "package_remote_sensing_hf.py",
    "crots_algorithm.py",
    "diagnose_crots_remote.py",
)
PACKAGED_PATHS: set[str] = set()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy(relative: str) -> None:
    source, target = ROOT / relative, OUT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    PACKAGED_PATHS.add(relative)


def main() -> None:
    PACKAGED_PATHS.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    for cache in OUT.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache)
    shutil.copy2(ROOT / "REMOTE_SENSING_HF_BUNDLE_README.md", OUT / "README.md")
    for relative in ROOT_FILES:
        copy(relative)
    for name in SCRIPT_FILES:
        copy(f"scripts/{name}")
    for directory in ("core", "datasets"):
        for source in (ROOT / directory).rglob("*.py"):
            copy(source.relative_to(ROOT).as_posix())
    for source in (DATA / "manifests").glob("*.csv"):
        copy(source.relative_to(ROOT).as_posix())
    copy("data/remote_sensing/dgle/metadata.json")
    copy("data/remote_sensing/dgle/source_val_metadata.json")
    copy("pretrain/resnet101-5d3b4d8f.pth")

    archives = []
    archive_dir = OUT / "archives"
    archive_dir.mkdir(exist_ok=True)
    for domain, split in SPLITS:
        source_dir = DATA / domain / split
        files = sorted(path for path in source_dir.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError(f"Empty split: {source_dir}")
        name = f"{domain}_{split}.zip"
        output = archive_dir / name
        print(f"Packing {domain}/{split}: {len(files)} files -> {output}", flush=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for source in files:
                archive.write(source, source.relative_to(ROOT).as_posix())
        archives.append({"path": f"archives/{name}", "files": len(files),
                         "bytes": output.stat().st_size, "sha256": digest(output)})
        PACKAGED_PATHS.add(f"archives/{name}")

    artifacts = []
    # Build from this invocation's explicit inputs, excluding old local results.
    for relative in sorted(PACKAGED_PATHS):
        source = OUT / relative
        artifacts.append({"path": relative,
                          "bytes": source.stat().st_size, "sha256": digest(source)})
    manifest = {"format": 1, "source": "DGLE remote sensing prepared splits",
                "split_counts": {f"{a}/{b}": next(x["files"] for x in archives
                                  if x["path"] == f"archives/{a}_{b}.zip")
                                 for a, b in SPLITS},
                "archives": archives, "artifacts": artifacts}
    (OUT / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Bundle ready: {OUT} ({len(artifacts)} artifacts)", flush=True)


if __name__ == "__main__":
    main()
