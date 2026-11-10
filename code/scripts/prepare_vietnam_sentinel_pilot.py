"""Prepare a small, unlabeled Sentinel-2 Collection 1 pilot for manual review.

Run with artifacts/sentinel_pilot_runtime/Scripts/python.exe.
Search metadata is prepared by artifacts/sentinel_pilot_probe.py.
"""

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data/vietnam_woody_cover_pilot_20261017'
REGIONS = {'quang_tri': [106.95, 16.72, 107.08, 16.85],
           'dong_nai': [107.08, 11.10, 107.21, 11.23]}
BANDS10 = ['blue', 'green', 'red', 'nir']
BANDS20 = ['rededge1', 'rededge2', 'rededge3', 'nir08', 'swir16', 'swir22']
NODATA = -9999.0
SIZE = 128


def choose_pair(before, after, bbox):
    def contains(item):
        b = item['bbox']
        return b[0] <= bbox[0] and b[1] <= bbox[1] and b[2] >= bbox[2] and b[3] >= bbox[3]
    pairs = []
    for a in before:
        for b in after:
            if not contains(a) or not contains(b) or a['properties']['grid:code'] != b['properties']['grid:code']:
                continue
            da = datetime.fromisoformat(a['properties']['datetime'].replace('Z', '+00:00'))
            db = datetime.fromisoformat(b['properties']['datetime'].replace('Z', '+00:00'))
            gap = abs((da.replace(year=2000) - db.replace(year=2000)).days)
            if gap <= 21:
                score = a['properties']['eo:cloud_cover'] + b['properties']['eo:cloud_cover'] + gap * 0.1
                pairs.append((score, a['id'], b['id'], a, b))
    if not pairs:
        raise RuntimeError('No covering same-tile pair within 21 calendar days; review search windows.')
    _, _, _, a, b = min(pairs, key=lambda p: p[:3])
    return a, b


def reflectance(raw, band):
    result = raw.astype('float32') * band['scale'] + band.get('offset', 0)
    result[raw == band.get('nodata', 0)] = NODATA
    return result


def write_tif(path, array, transform, crs, names):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.part.tif')
    with rasterio.open(partial, 'w', driver='GTiff', height=array.shape[1], width=array.shape[2],
                       count=array.shape[0], dtype=str(array.dtype), transform=transform, crs=crs,
                       nodata=0 if array.dtype == np.uint8 else NODATA, compress='deflate') as dst:
        dst.write(array)
        for i, name in enumerate(names, 1):
            dst.set_band_description(i, name)
    partial.replace(path)


def region_grid(item, bbox):
    asset = item['assets']['red']
    tr = rasterio.Affine(*asset['proj:transform'][:6])
    crs = rasterio.CRS.from_epsg(item['properties']['proj:epsg'])
    projected = transform_bounds('EPSG:4326', crs, *bbox, densify_pts=21)
    window = from_bounds(*projected, transform=tr)
    col = math.floor(window.col_off / 2) * 2
    row = math.floor(window.row_off / 2) * 2
    end_col = math.ceil((window.col_off + window.width) / 2) * 2
    end_row = math.ceil((window.row_off + window.height) / 2) * 2
    window = Window(col, row, end_col - col, end_row - row)
    return window_bounds(window, tr), crs


def fetch_band(item, key, bbox_projected):
    asset = item['assets'][key]
    url = asset['href']
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com':
        raise ValueError(f'Unexpected public data host: {parsed.hostname}')
    print(f"  {item['id']} {key}", flush=True)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='40',
                      GDAL_HTTP_MAX_RETRY='2', GDAL_HTTP_RETRY_DELAY='2',
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
        with rasterio.open(url) as src:
            w = from_bounds(*bbox_projected, transform=src.transform).round_offsets().round_lengths()
            if w.col_off < 0 or w.row_off < 0 or w.col_off + w.width > src.width or w.row_off + w.height > src.height:
                raise ValueError('Crop extends outside source tile')
            raw = src.read(1, window=w)
            tr = src.window_transform(w)
    if key == 'scl':
        return raw.astype('uint8'), tr
    # Collection 1 pixels retain the provider's scale/offset; apply exactly once.
    return reflectance(raw, asset['raster:bands'][0]), tr


def prepare_epoch(item, projected, crs, directory):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / 'source.json'
    if metadata.exists() and json.loads(metadata.read_text())['id'] != item['id']:
        raise RuntimeError('Existing source differs; use a new output directory.')
    metadata.write_text(json.dumps(item, indent=2), encoding='utf-8')
    for names, filename in [(BANDS10, 'sr10m.tif'), (BANDS20, 'sr20m.tif'), (['scl'], 'scl20m.tif')]:
        target = directory / filename
        if target.exists():
            with rasterio.open(target) as src:
                assert src.count == len(names) and src.crs == crs
            continue
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda key: fetch_band(item, key, projected), names))
        assert all(tr.almost_equals(results[0][1]) for _, tr in results)
        write_tif(target, np.stack([arr for arr, _ in results]), results[0][1], crs, names)


def preview(array, indices, maximum):
    selected = array[indices].transpose(1, 2, 0)
    valid = np.all(selected != NODATA, axis=-1)
    selected = np.clip(selected / maximum, 0, 1) ** (1 / 1.5)
    selected[~valid] = 0
    return Image.fromarray(np.uint8(selected * 255))


def save_epoch_site(region_dir, epoch, site_dir, row, col):
    output = site_dir / epoch
    for filename in ('sr10m.tif', 'sr20m.tif', 'scl20m.tif'):
        with rasterio.open(region_dir / epoch / filename) as src:
            factor = 1 if filename == 'sr10m.tif' else 2
            window = Window(col // factor, row // factor, SIZE // factor, SIZE // factor)
            data = src.read(window=window)
            write_tif(output / filename, data, src.window_transform(window), src.crs, src.descriptions)
            if filename == 'sr10m.tif':
                preview(data, [2, 1, 0], 0.3).save(output / 'rgb.png')
                preview(data, [3, 2, 1], 0.5).save(output / 'false_color.png')


def self_test():
    raw = np.array([0, 1000, 2500], dtype='uint16')
    got = reflectance(raw, {'scale': 0.0001, 'offset': -0.1, 'nodata': 0})
    assert got[0] == NODATA and abs(got[1]) < 1e-6 and abs(got[2] - 0.15) < 1e-6
    fake = lambda ident, date, tile='A', cloud=1: {'id': ident, 'bbox': [0, 0, 2, 2], 'properties': {'datetime': date, 'grid:code': tile, 'eo:cloud_cover': cloud}}
    a = fake('a', '2020-03-13T00:00:00Z')
    far = fake('far', '2024-06-13T00:00:00Z', cloud=0)
    good = fake('good', '2024-03-22T00:00:00Z')
    assert choose_pair([a], [far, good], [0.5, 0.5, 1.5, 1.5])[1]['id'] == 'good'
    print('Self-check passed: scale/offset/nodata and same-season pair selection.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    root = args.output.resolve()
    rows, footprints, manifests = [], [], []
    for region, bbox in REGIONS.items():
        searches = [json.loads((root / f'{region}_{y}_c1_search.json').read_text())['features'] for y in (2020, 2024)]
        before, after = choose_pair(*searches, bbox)
        projected, crs = region_grid(before, bbox)
        directory = root / 'region_cache' / region
        print(f"{region}: {before['id']} -> {after['id']}", flush=True)
        for epoch, item in [('before', before), ('after', after)]:
            prepare_epoch(item, projected, crs, directory / epoch)
        with rasterio.open(directory / 'before/sr10m.tif') as src:
            height, width, transform = src.height, src.width, src.transform
        with rasterio.open(directory / 'before/scl20m.tif') as src:
            scl0 = src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        with rasterio.open(directory / 'after/scl20m.tif') as src:
            scl1 = src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        candidates = []
        for row in np.linspace(0, height - SIZE, 6, dtype=int):
            for col in np.linspace(0, width - SIZE, 6, dtype=int):
                row, col = int(row // 2 * 2), int(col // 2 * 2)
                a, b = scl0[row:row+SIZE, col:col+SIZE], scl1[row:row+SIZE, col:col+SIZE]
                usable = np.isin(a, [4, 5, 6]) & np.isin(b, [4, 5, 6])
                quality, vegetation = float(usable.mean()), float((a == 4).mean())
                if quality >= 0.98 and vegetation >= 0.5:
                    candidates.append((row, col, quality, vegetation))
        # shortcut: two scene pairs are a labeling pilot, not a representative national benchmark.
        rng = np.random.default_rng(20261017)
        rng.shuffle(candidates)
        selected = candidates[:10]
        if not selected:
            raise RuntimeError(f'No usable candidate chips in {region}; inspect source selection.')
        for number, (row, col, quality, vegetation) in enumerate(selected, 1):
            site_id = f'{region}_{number:02d}'
            site_dir = root / 'sites' / site_id
            for epoch in ('before', 'after'):
                save_epoch_site(directory, epoch, site_dir, row, col)
            window = Window(col, row, SIZE, SIZE)
            tr = rasterio.windows.transform(window, transform)
            b = window_bounds(window, transform)
            wgs = transform_bounds(crs, 'EPSG:4326', *b, densify_pts=21)
            metadata = {'site_id': site_id, 'region': region, 'width': SIZE, 'height': SIZE,
                        'before_date': before['properties']['datetime'][:10], 'after_date': after['properties']['datetime'][:10],
                        'before_scene': before['id'], 'after_scene': after['id'],
                        'crs': crs.to_string(), 'transform': list(tr)[:6], 'bbox_wgs84': list(wgs),
                        'scl_usable_fraction_both': quality, 'scl_vegetation_fraction_before': vegetation,
                        'status': 'unlabeled', 'selection': 'fixed-grid shuffled candidates; SCL quality and initial vegetation only; not verified woody cover or loss'}
            (site_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            manifests.append(metadata)
            rows.append({'site_id': site_id, 'before_date': metadata['before_date'], 'after_date': metadata['after_date'],
                         'review_status': 'pending', 'image_usable': '', 'woody_cover_confirmed': '', 'loss_present': '', 'notes': ''})
            west, south, east, north = wgs
            footprints.append({'type': 'Feature', 'properties': metadata, 'geometry': {'type': 'Polygon', 'coordinates': [[[west,south],[east,south],[east,north],[west,north],[west,south]]]}})
            print(f'  saved {site_id}: usable={quality:.3f}', flush=True)
    (root / 'manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')
    (root / 'site_footprints.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': footprints}, indent=2), encoding='utf-8')
    review_file = root / 'review.csv'
    if not review_file.exists():
        with review_file.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (root / 'run_summary.json').write_text(json.dumps({'pairs_downloaded': len(manifests), 'labels_created': 0,
        'collection': 'sentinel-2-c1-l2a', 'reflectance': 'float32 DN*scale+offset from source STAC; nodata=-9999',
        'native_resolution_preserved': {'sr10m': BANDS10, 'sr20m': BANDS20, 'scl20m': ['scl']},
        'quality_note': 'SCL screen is automated; manual cloud, registration, woody-cover and change review still required.'}, indent=2), encoding='utf-8')
    print(f'Complete: {len(manifests)} unlabeled candidate pairs.', flush=True)


if __name__ == '__main__':
    main()
