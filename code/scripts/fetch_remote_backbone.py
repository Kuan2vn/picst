"""Fetch the ResNet101 ImageNet initialization used by this repository.

Run once on a machine with access to download.pytorch.org, then copy the file
to the same relative path on the training host. The filename embeds the first
eight hexadecimal digits of its SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


URL = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
EXPECTED_PREFIX = "5d3b4d8f"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "pretrain" / "resnet101-5d3b4d8f.pth"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        value = digest(output)
        if not value.startswith(EXPECTED_PREFIX):
            raise RuntimeError(f"Existing checkpoint has unexpected SHA-256: {output}: {value}")
        print(f"Verified existing {output}: {value}")
        return
    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(URL, timeout=90) as source, temporary.open("wb") as target:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                target.write(chunk)
        value = digest(temporary)
        if not value.startswith(EXPECTED_PREFIX):
            raise RuntimeError(f"Downloaded checkpoint SHA-256 mismatch: {value}")
        os.replace(temporary, output)
        print(f"Saved {output}: {value}")
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
