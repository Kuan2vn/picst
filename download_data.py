#!/usr/bin/env python3
"""Download the private data bundle from Hugging Face and unpack it into ./data (needs `huggingface-cli login`
with an account that was granted access to the private dataset)."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = 'kuan2/picst-rs-sfda-data'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'data')
    ap.add_argument('--only', nargs='*', help='subset of files from SHA256SUMS.json (default: all)')
    a = ap.parse_args()
    a.data_dir.mkdir(parents=True, exist_ok=True)
    sums = json.loads(Path(hf_hub_download(REPO, 'SHA256SUMS.json', repo_type='dataset')).read_text())
    for name, info in sums.items():
        if a.only and name not in a.only:
            continue
        path = Path(hf_hub_download(REPO, name, repo_type='dataset'))
        if sha256(path) != info['sha256']:
            raise RuntimeError(f'Checksum mismatch for {name}')
        if name.endswith('.zip'):
            marker = a.data_dir/f'.{name}.extracted'
            if not marker.exists():
                with zipfile.ZipFile(path) as z:
                    for member in z.infolist():  # refuse paths that escape the data directory
                        if not (a.data_dir/member.filename).resolve().is_relative_to(a.data_dir.resolve()):
                            raise RuntimeError(f'Unsafe path in {name}: {member.filename}')
                    z.extractall(a.data_dir)
                marker.write_text(info['sha256'])
        else:
            target = a.data_dir/name
            if not target.exists() or sha256(target) != info['sha256']:
                target.write_bytes(path.read_bytes())
        print('ok', name)
    print('Data ready in', a.data_dir)


if __name__ == '__main__':
    main()
