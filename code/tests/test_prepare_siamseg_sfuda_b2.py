"""Check the exported tile split, patch pairs and label-free target manifest."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from scripts.prepare_dgle_remote_sensing import color_mask_to_ids
from scripts.prepare_siamseg_data import tile_members


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/remote_sensing/siamseg_sfuda_b2_v1'


def test_prepared():
    config = json.loads((ROOT/'configs/remote_sensing/siamseg_sfuda_b2_v1.json').read_text())
    metadata = json.loads((DATA/'metadata.json').read_text())
    assert metadata['config_sha256'] == hashlib.sha256((ROOT/'configs/remote_sensing/siamseg_sfuda_b2_v1.json').read_bytes()).hexdigest()
    assert metadata['target_train_labels'] is False
    expected = {'source_train': (1046, set(map(str, config['source_train_tiles']))),
                'source_val': (1208, set(map(str, config['source_val_tiles']))),
                'target_train': (3456, set(config['target_train_tiles'])),
                'target_eval': (2016, set(config['target_eval_tiles']))}
    for split, (count, tiles) in expected.items():
        with (DATA/'manifests'/f'{split}.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
        assert len({row['image'] for row in rows}) == count
        observed = set()
        for row in rows:
            image = DATA/row['image']
            assert image.is_file()
            stem = image.stem.split('_')
            tile = stem[0] if split.startswith('source') else '_'.join(stem[:2])
            observed.add(tile)
            assert row['label'] == '' if split == 'target_train' else row['label'] != ''
            if row['label']:
                label = DATA/row['label']
                assert label.is_file() and label.stem == image.stem
        assert observed == tiles
        for row in rows[::max(1, len(rows)//8)]:
            with Image.open(DATA/row['image']) as image:
                assert image.mode == 'RGB' and image.size == (512, 512)
            if row['label']:
                with Image.open(DATA/row['label']) as label:
                    values = set(np.unique(np.asarray(label)).tolist())
                    assert label.mode == 'L' and label.size == (512, 512)
                    assert values <= {0, 1, 2, 3, 4, 5, 255}
    assert not any((DATA/'target_train').rglob('labels/*'))

    # Compare one exported patch from each split against its raw tile and mask.
    raw = ROOT/'data/remote_sensing'
    archives = {
        'Vaihingen': ('ISPRS_semantic_labeling_Vaihingen.zip',
                      'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip'),
        'Potsdam': ('2_Ortho_RGB.zip', '5_Labels_all_noBoundary.zip'),
    }
    for domain, (image_zip, label_zip) in archives.items():
        image_path, label_path = raw/domain/image_zip, raw/domain/label_zip
        if not image_path.is_file() or not label_path.is_file():
            continue  # Raw archives are optional after export, including on HPC.
        with zipfile.ZipFile(image_path) as images, zipfile.ZipFile(label_path) as labels:
            image_members = tile_members(images, domain, 'image')
            label_members = tile_members(labels, domain, 'label')
            splits = ('source_train', 'source_val') if domain == 'Vaihingen' else ('target_train', 'target_eval')
            for split in splits:
                with (DATA/'manifests'/f'{split}.csv').open(newline='', encoding='utf-8') as stream:
                    row = next(csv.DictReader(stream))
                stem = Path(row['image']).stem.split('_')
                tile, x, y = ((stem[0], *map(int, stem[1:])) if domain == 'Vaihingen'
                              else ('_'.join(stem[:2]), *map(int, stem[2:])))
                box = (x, y, x+512, y+512)
                with images.open(image_members[tile]) as source, Image.open(source) as image:
                    expected = np.asarray(image.convert('RGB').crop(box))
                with Image.open(DATA/row['image']) as exported:
                    assert np.array_equal(np.asarray(exported), expected)
                if row['label']:
                    with labels.open(label_members[tile]) as source, Image.open(source) as label:
                        expected = np.asarray(color_mask_to_ids(label).crop(box))
                    with Image.open(DATA/row['label']) as exported:
                        assert np.array_equal(np.asarray(exported), expected)


if __name__ == '__main__':
    test_prepared()
    print('Dataset manifests, tile disjointness, image/mask pairs and labels OK')
