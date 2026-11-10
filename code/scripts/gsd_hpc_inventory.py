"""Read the existing source/warm provenance before starting a separate GSD study."""
import json
from pathlib import Path
import torch


def main():
    root = Path.home() / "Tu/hf_remote_sensing_bundle"
    base = root / "results_remote_sensing/siamseg_sfuda_b2_v1/vh2pd"
    paths = [base / "source/seed1/last.pth", base / "snd/seed1/warm/last.pth"]
    records = []
    for path in paths:
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=False)
            item.update({key: state.get(key) for key in (
                "iteration", "stage", "architecture", "protocol", "checkpoint_selection",
                "source_manifest_sha256", "source_checkpoint_sha256", "protocol_signature",
                "training_settings", "smoke_only")})
            item["bytes"] = path.stat().st_size
            del state
        records.append(item)
    archives = root / "data/remote_sensing/isprs/original/Potsdam/Potsdam"
    print(json.dumps({"torch": str(torch.__version__), "checkpoints": records,
        "raw_archives": [{"path": str(p), "bytes": p.stat().st_size}
                         for p in archives.glob("*.zip")],
        "pilot_irrg_exists": (root / "data/remote_sensing/vh2pd_irrg_pilot_20261010/METADATA.json").is_file()
    }, indent=2))


if __name__ == "__main__":
    main()
