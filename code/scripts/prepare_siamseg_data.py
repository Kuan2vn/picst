"""Separate native-mask datasets for the public SiamSeg UDA code.

LoveDA is extracted byte-for-byte. ISPRS follows the public tile lists with
the converter's stride/overwrite defects repaired explicitly. It is not the
DGLE split or a verified reconstruction of SiamSeg's unpublished patch lists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

import numpy as np
from PIL import Image

from scripts.prepare_dgle_remote_sensing import color_mask_to_ids, write_manifest

ROOT = Path(__file__).resolve().parents[1]
SFUDA_CONFIG = ROOT / 'configs/remote_sensing/siamseg_sfuda_b2_v1.json'
VAI_TRAIN = (1, 11, 13, 15, 17, 21, 23, 26, 28, 3, 30, 32, 34, 37, 5, 7)
PD_TRAIN = tuple(f'{i}_{j}' for i, js in (
    (2, (10, 11, 12)), (3, (10, 11, 12)), (4, (10, 11, 12)),
    (5, (10, 11, 12)), (6, (7, 8, 9, 10, 11, 12)),
    (7, (7, 8, 9, 10, 11, 12))) for j in js)
PD_VAL = ('5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14',
          '2_13', '4_15', '2_14', '5_13', '4_13', '3_14', '7_13')
COLORS = ((255, 255, 255), (0, 0, 255), (0, 255, 255),
          (0, 255, 0), (255, 255, 0), (255, 0, 0))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def positions(length, crop, stride):
    if length < crop:
        raise ValueError('Native tile smaller than crop')
    return sorted({min(index * stride, length - crop)
                   for index in range(math.ceil((length - crop) / stride) + 1)})


def native_mask(rgb):
    values = np.asarray(rgb.convert('RGB'))
    mask = np.zeros(values.shape[:2], dtype=np.uint8)
    for label, color in enumerate(COLORS, 1):
        mask[np.all(values == color, axis=2)] = label
    if np.any((mask == 0) & np.any(values != 0, axis=2)):
        raise ValueError('Unknown ISPRS label color')
    return Image.fromarray(mask)


def extract_loveda(raw, output):
    counts, archives = {}, {}
    specs = (('Train', 'Rural', 1366), ('Train', 'Urban', 1156), ('Val', 'Urban', 677))
    for split, domain, expected in specs:
        path = raw / f'{split}.zip'
        archives[path.name] = sha256(path)
        with zipfile.ZipFile(path) as archive:
            for kind in ('images_png', 'masks_png'):
                members = sorted(name for name in archive.namelist()
                                 if PurePosixPath(name).parts[:3] == (split, domain, kind)
                                 and len(PurePosixPath(name).parts) == 4 and name.endswith('.png'))
                if len(members) != expected:
                    raise ValueError(f'{split}/{domain}/{kind}: {len(members)} != {expected}')
                for name in members:
                    destination = output / 'LoveDA' / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # Always copy original masks: MMSeg applies reduce_zero_label once.
                    with archive.open(name) as src, destination.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                counts[f'{split}/{domain}/{kind}'] = len(members)
        print(f'LoveDA {split}/{domain}: {expected} image/mask pairs', flush=True)
    return {'counts': counts, 'raw_archives_sha256': archives,
            'mask_encoding': 'native 0 ignore; 1..7 classes; MMSeg reduces once',
            'preprocess': 'original PNG bytes; same split as public config',
            'target_train_masks_present': True,
            'target_mask_usage': 'upstream LoadAnnotations + RandomCrop(cat_max_ratio=0.75); not label-free preprocessing'}


def tile_members(archive, dataset, role):
    result = {}
    for name in archive.namelist():
        parts = PurePosixPath(name).parts
        if dataset == 'Vaihingen' and role == 'image' and (not parts or parts[0] != 'top'):
            continue
        if not name.lower().endswith('.tif'):
            continue
        pattern = r'area(\d+)' if dataset == 'Vaihingen' else r'top_potsdam_(\d+_\d+)'
        found = re.search(pattern, parts[-1], re.I)
        if found:
            key = found[1]
            if key in result:
                raise ValueError(f'Duplicate tile {key} in {archive.filename}')
            result[key] = name
    return result


def extract_isprs(raw, output):
    counts, archives = {}, {}
    specs = (
        ('Vaihingen', 'ISPRS_semantic_labeling_Vaihingen.zip',
         'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip',
         'Vaihingen_IRRG_DA', tuple(str(x) for x in VAI_TRAIN), (), 256),
        ('Potsdam', '2_Ortho_RGB.zip', '5_Labels_all_noBoundary.zip',
         'Potsdam_RGB_DA', PD_TRAIN, PD_VAL, 512),
    )
    for dataset, image_name, label_name, folder, train, val, stride in specs:
        image_path, label_path = raw / dataset / image_name, raw / dataset / label_name
        for path in (image_path, label_path):
            archives[f'{dataset}/{path.name}'] = sha256(path)
        with zipfile.ZipFile(image_path) as images, zipfile.ZipFile(label_path) as labels:
            im = tile_members(images, dataset, 'image')
            lab = tile_members(labels, dataset, 'label')
            splits = {'train': train, 'val': val}
            for split, tiles in splits.items():
                if not tiles:
                    continue
                names = []
                for tile in tiles:
                    if tile not in im or tile not in lab:
                        raise ValueError(f'{dataset}/{split}: missing tile image or mask {tile}')
                    with images.open(im[tile]) as stream, Image.open(stream) as src:
                        image = src.convert('RGB')
                    with labels.open(lab[tile]) as stream, Image.open(stream) as src:
                        mask = native_mask(src)
                    if image.size != mask.size:
                        raise ValueError(f'{dataset}/{tile}: image/mask shape mismatch')
                    for y in positions(image.height, 512, stride):
                        for x in positions(image.width, 512, stride):
                            prefix = f'area{tile}' if dataset == 'Vaihingen' else tile
                            stem = f'{prefix}_{x}_{y}_{x + 512}_{y + 512}'
                            for kind, source in (('img_dir', image), ('ann_dir', mask)):
                                destination = output / folder / kind / split / f'{stem}.png'
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                source.crop((x, y, x + 512, y + 512)).save(destination)
                            names.append(stem)
                    print(f'{dataset}/{split}: tile {tile}; patches so far {len(names)}', flush=True)
                (output / folder / f'{split}.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
                counts[f'{folder}/{split}'] = len(names)
    return {'counts': counts, 'raw_archives_sha256': archives,
            'tiles': {'vai_train': list(VAI_TRAIN), 'pd_train': list(PD_TRAIN), 'pd_val': list(PD_VAL)},
            'mask_encoding': 'native 0 ignore; 1..6 classes; MMSeg reduces once',
            'preprocess': 'public converter tile groups; stride repaired and unique border coordinates',
            'patch_lists_status': 'generated; authors train.txt/val.txt absent from public repo',
            'published_counts': {'Vaihingen_IRRG_DA/train': 1296,
                                 'Potsdam_RGB_DA/train': 2904, 'Potsdam_RGB_DA/val': 1694},
            'paper_protocol_verified': False}


def prepare_sfuda(raw, output, config_path=SFUDA_CONFIG):
    """Export the fixed tile-disjoint, label-free-target B2 protocol."""
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config['protocol'] != 'siamseg_sfuda_b2_v1':
        raise ValueError('Unexpected protocol config')
    splits = (
        ('Vaihingen', 'source_train', config['source_train_tiles'], config['source_stride'], True),
        ('Vaihingen', 'source_val', config['source_val_tiles'], config['source_stride'], True),
        ('Potsdam', 'target_train', config['target_train_tiles'], config['target_stride'], False),
        ('Potsdam', 'target_eval', config['target_eval_tiles'], config['target_stride'], True),
    )
    if set(config['source_train_tiles']) & set(config['source_val_tiles']):
        raise ValueError('Source tile overlap')
    if set(config['target_train_tiles']) & set(config['target_eval_tiles']):
        raise ValueError('Target tile overlap')
    archives = {}
    coordinates = {}
    counts = {}
    for domain in ('Vaihingen', 'Potsdam'):
        names = (('ISPRS_semantic_labeling_Vaihingen.zip',
                  'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip')
                 if domain == 'Vaihingen' else
                 ('2_Ortho_RGB.zip', '5_Labels_all_noBoundary.zip'))
        image_path, label_path = (raw / domain / name for name in names)
        archives[f'{domain}/{names[0]}'] = sha256(image_path)
        archives[f'{domain}/{names[1]}'] = sha256(label_path)
        with zipfile.ZipFile(image_path) as images, zipfile.ZipFile(label_path) as labels:
            image_members = tile_members(images, domain, 'image')
            label_members = tile_members(labels, domain, 'label')
            for _, split, tiles, stride, labeled in (s for s in splits if s[0] == domain):
                if not set(map(str, tiles)) <= set(image_members):
                    raise ValueError(f'{split}: missing image tile')
                if labeled and not set(map(str, tiles)) <= set(label_members):
                    raise ValueError(f'{split}: missing label tile')
                rows, geometry = [], []
                for tile in tiles:
                    key = str(tile)
                    with images.open(image_members[key]) as stream, Image.open(stream) as src:
                        image = src.convert('RGB')
                    mask = None
                    if labeled:
                        with labels.open(label_members[key]) as stream, Image.open(stream) as src:
                            mask = color_mask_to_ids(src)
                        if image.size != mask.size:
                            raise ValueError(f'{domain}/{key}: image/mask dimensions differ')
                    for y in positions(image.height, config['crop'], stride):
                        for x in positions(image.width, config['crop'], stride):
                            stem = f'{key}_{x}_{y}'
                            rel_image = f'{split}/images/{stem}.png'
                            image_out = output / rel_image
                            image_out.parent.mkdir(parents=True, exist_ok=True)
                            image.crop((x, y, x+config['crop'], y+config['crop'])).save(image_out)
                            rel_mask = ''
                            if mask is not None:
                                rel_mask = f'{split}/labels/{stem}.png'
                                mask_out = output / rel_mask
                                mask_out.parent.mkdir(parents=True, exist_ok=True)
                                mask.crop((x, y, x+config['crop'], y+config['crop'])).save(mask_out)
                            rows.append((rel_image, rel_mask))
                            geometry.append((key, x, y, x+config['crop'], y+config['crop']))
                    print(f'{split}: {key}, {len(rows)} patches', flush=True)
                manifest = output / 'manifests' / f'{split}.csv'
                write_manifest(manifest, rows)
                counts[split] = len(rows)
                coordinates[split] = hashlib.sha256(json.dumps(geometry, separators=(',', ':')).encode()).hexdigest()
    expected = {'source_train': 1046, 'source_val': 1208, 'target_train': 3456, 'target_eval': 2016}
    if counts != expected:
        raise ValueError(f'Unexpected patch counts: {counts} != {expected}')
    inventory = output / 'file_sha256.tsv'
    with inventory.open('w', encoding='utf-8', newline='\n') as stream:
        for path in sorted((*output.rglob('*.png'), *output.glob('manifests/*.csv'))):
            stream.write(f'{sha256(path)}\t{path.relative_to(output).as_posix()}\n')
    metadata = {'protocol': config['protocol'], 'config_sha256': sha256(config_path),
                'preparer_sha256': sha256(__file__), 'raw_archives_sha256': archives,
                'coordinate_sha256': coordinates, 'counts': counts,
                'file_inventory_sha256': sha256(inventory),
                'manifest_sha256': {split: sha256(output/'manifests'/f'{split}.csv') for split in counts},
                'target_train_labels': False}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('r2u', 'vh2pd', 'sfuda_b2'), required=True)
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/remote_sensing')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/remote_sensing/siamseg_official')
    args = parser.parse_args()
    if args.task == 'sfuda_b2':
        args.output = ROOT / 'data/remote_sensing/siamseg_sfuda_b2_v1' if args.output == ROOT / 'data/remote_sensing/siamseg_official' else args.output
    args.output.mkdir(parents=True, exist_ok=True)
    report = (extract_loveda(args.raw_root / 'LoveDA/raw', args.output) if args.task == 'r2u'
              else prepare_sfuda(args.raw_root / 'isprs/raw', args.output) if args.task == 'sfuda_b2'
              else extract_isprs(args.raw_root / 'isprs/raw', args.output))
    report.update(task=args.task, preparer_sha256=sha256(__file__))
    (args.output / f'preprocess_{args.task}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
