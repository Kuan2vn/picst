"""Export inference-only checkpoints for the authorized public research bundle."""
import hashlib
import json
from pathlib import Path
import torch


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    old = Path.home() / "Tu/hf_remote_sensing_bundle"
    output = Path.home() / "Tu/gsd_sfuda_20261012/public_checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_08500.pth": Path.home() / "Tu/gsd_sfuda_20261012/checkpoints/source_resume_08500.pth",
        "source_smoke.pth": old / "results_remote_sensing/siamseg_sfuda_b2_v1/vh2pd/smoke/source/last.pth"}
    fields = ("task", "stage", "iteration", "architecture", "protocol", "protocol_sha256",
              "source_split", "source_manifest_sha256", "source_val_manifest_sha256",
              "source_checkpoint_sha256", "num_classes", "normalization", "train_policy",
              "checkpoint_selection", "training_settings", "smoke_only", "seed")
    records = []
    for name, source in paths.items():
        state = torch.load(source, map_location="cpu", weights_only=False)
        clean = {key: state[key] for key in fields if key in state}
        clean.update(feature_extractor=state["feature_extractor"], classifier=state["classifier"],
                     original_checkpoint_sha256=digest(source))
        path = output / name
        torch.save(clean, path)
        records.append({"file": name, "sha256": digest(path), "bytes": path.stat().st_size,
                        "iteration": clean["iteration"], "stage": clean["stage"],
                        "smoke_only": clean["smoke_only"], "optimizer_included": False})
        del state, clean
    (output / "CHECKPOINTS.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
