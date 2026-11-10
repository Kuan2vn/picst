"""Verify the uploaded public DGLE bundle against the local staging folder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "hf_remote_sensing_bundle"
REPO = "kuan2/snd-remask-dgle-remote-sensing-hpc"


def main() -> None:
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", token=False)
    if info.private:
        raise AssertionError(f"Repository is not public: {REPO}")
    manifest = json.loads((STAGING / "bundle_manifest.json").read_text(encoding="utf-8"))
    published_paths = {item["path"] for item in manifest["artifacts"]}
    published_paths.update(("README.md", "bundle_manifest.json"))
    local = {name: STAGING / name for name in published_paths}
    remote = {item.path: item for item in api.list_repo_tree(
        REPO, repo_type="dataset", recursive=True, expand=True, token=False)
              if isinstance(item, RepoFile) and item.path != ".gitattributes"}
    if local.keys() != remote.keys():
        raise AssertionError({"missing": sorted(local.keys() - remote.keys()),
                              "extra": sorted(remote.keys() - local.keys())})
    digests = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    hash_verified = 0
    for name, path in local.items():
        entry = remote[name]
        if entry.size != path.stat().st_size:
            raise AssertionError(f"Size mismatch: {name}: {entry.size} != {path.stat().st_size}")
        if entry.lfs is not None and name in digests:
            if entry.lfs.sha256 != digests[name]:
                raise AssertionError(f"SHA-256 mismatch: {name}")
            hash_verified += 1
    downloaded = Path(hf_hub_download(
        REPO, "bundle_manifest.json", repo_type="dataset",
        local_dir=ROOT / "tmp" / "hf_remote_verify", token=False))
    if hashlib.sha256(downloaded.read_bytes()).digest() != hashlib.sha256(
        (STAGING / "bundle_manifest.json").read_bytes()).digest():
        raise AssertionError("Remote manifest differs from local manifest")
    print(f"Public repo verified without authentication: {REPO}")
    print(f"Files and sizes verified: {len(local)}")
    print(f"LFS SHA-256 verified: {hash_verified}")
    print("Remote bundle manifest byte-for-byte verified")


if __name__ == "__main__":
    main()
