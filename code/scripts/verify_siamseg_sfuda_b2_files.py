"""Verify every exported B2 patch after transfer; reads no checkpoint tensors."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, weights):
    config_path = Path('configs/remote_sensing/siamseg_sfuda_b2_v1.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    metadata = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if metadata['config_sha256'] != sha256(config_path) or metadata['protocol'] != config['protocol']:
        raise ValueError('Dataset/config mismatch')
    if sha256(weights) != config['pretrain_sha256']:
        raise ValueError('ImageNet MiT-B2 checkpoint mismatch')
    inventory = root/'file_sha256.tsv'
    if sha256(inventory) != metadata['file_inventory_sha256']:
        raise ValueError('Dataset inventory checksum mismatch')
    count = 0
    for line in inventory.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('\t', 1)
        path = (root/name).resolve()
        if root.resolve() not in path.parents or sha256(path) != expected:
            raise ValueError(f'Dataset file mismatch: {name}')
        count += 1
    print(f'Verified {count} dataset files and MiT-B2 weights')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/remote_sensing/siamseg_sfuda_b2_v1'))
    parser.add_argument('--weights', type=Path, default=Path('pretrain/mit_b2_20220624-66e8bf70.pth'))
    args = parser.parse_args()
    verify(args.data_root, args.weights)
