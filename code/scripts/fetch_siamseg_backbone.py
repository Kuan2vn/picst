"""Fetch an official MMSeg converted ImageNet MiT backbone."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URLS = {
    'b2': ('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth', '66e8bf70'),
    'b5': ('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth', '658746d9'),
}
FULL_SHA256 = {'b2': '66e8bf7068891148bc097433ddf8e55e690b6f9e17f543edb405f1112b8d330e'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=URLS, default='b5')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    partial = args.output.with_suffix('.download')
    digest = hashlib.sha256()
    url, prefix = URLS[args.model]
    with urllib.request.urlopen(url, timeout=60) as src, partial.open('wb') as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
            dst.write(block)
            digest.update(block)
    actual = digest.hexdigest()
    if not actual.startswith(prefix) or (args.model in FULL_SHA256 and actual != FULL_SHA256[args.model]):
        raise ValueError(f'Official filename SHA-256 prefix mismatch: {actual}')
    partial.replace(args.output)
    print(f'{args.output}: SHA-256 {actual}; standard MMSeg pretrain, authors file identity not confirmed')


if __name__ == '__main__':
    main()
