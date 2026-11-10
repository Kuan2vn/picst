"""Build an explicit code snapshot for both HPC and the public Kaggle bundle."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("train_SND.py", "train_SND_remote.py", "train_DTST_remote.py", "train_SRMP.py", "train_SRMP_remote.py",
              "train_ReMask.py", "configs/snd_vh2pd_b2.json",
              "configs/remote_sensing/siamseg_sfuda_b2_v1.json", "references/siamseg/LICENSE",
              "references/segformer/LICENSE")
SCRIPTS = ("__init__.py", "snd_remote.py", "remote_sensing_transforms.py", "remote_sensing_results.py",
           "diagnose_crots_remote.py", "prepare_dgle_remote_sensing.py",
           "gsd_reliability.py", "dtst_remote.py", "prepare_gsd_views.py", "diagnose_gsd_reliability.py",
           "package_gsd_code.py", "dtst_study_report.py")


def build(output):
    paths = [ROOT / p for p in ROOT_FILES]
    paths += [ROOT / "scripts" / p for p in SCRIPTS]
    for directory in ("core", "datasets", "configs/gsd_sfuda", "configs/dtst_sfuda"):
        paths += [p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in (".py", ".json")]
    paths += [ROOT / "tests/test_gsd_reliability.py", ROOT / "tests/test_snd_b2_alignment.py", ROOT / "tests/test_dtst_remote.py"]
    for optional in ("run_gsd_study_hpc.sh", "run_dtst_study_hpc.sh", "requirements_gsd_kaggle.txt"):
        if (ROOT / optional).is_file():
            paths.append(ROOT / optional)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            content = path.read_bytes().replace(b"\r\n", b"\n")
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, content)
            records.append({"path": relative, "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest()})
        archive.writestr("code_snapshot_manifest.json", json.dumps(records, indent=2) + "\n")
    print(json.dumps({"archive": str(output), "files": len(records), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/gsd_code_20261012.zip")
    build(parser.parse_args().output)
