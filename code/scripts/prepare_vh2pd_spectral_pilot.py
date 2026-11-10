"""Export genuine official Potsdam IRRG at the frozen RGB pilot coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/training_free_subspace_20261008'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): value.update(block)
    return value.hexdigest()


def main():
    parent=REPORT/'graph_vh_sourceval/PROTOCOL.json'
    protocol=json.loads(parent.read_text(encoding='utf-8'))['locked']
    names={role:protocol[role+'_images'] for role in ('fit','calibration','screen')}
    if set(names['fit'])&set(names['calibration']): raise ValueError('Fit/cal overlap')
    selected=names['fit']+names['calibration']+names['screen']
    if len(selected)!=208 or len(set(selected))!=208: raise ValueError('Unexpected pilot coordinates')
    rgb=ROOT/'data/remote_sensing/siamseg_sfuda_b2_v1'
    archive_path=ROOT/'data/remote_sensing/isprs/original/Potsdam/Potsdam/3_Ortho_IRRG.zip'
    output=ROOT/'data/remote_sensing/vh2pd_irrg_pilot_20261010'
    output.mkdir(parents=True,exist_ok=False)
    metadata={'roles':names,'parent_protocol_sha256':sha(parent),'raw_archive':'3_Ortho_IRRG.zip',
        'raw_archive_sha256':sha(archive_path),'channel_policy':'Official IRRG archive stored planes; no hand permutation or synthetic NIR',
        'rgb_reference_sha256':{},'files_sha256':{},'tiles':{},'target_train_gt_opened':0}
    tiles={}
    for name in selected:
        found=re.fullmatch(r'(\d+_\d+)_(\d+)_(\d+)\.png',Path(name).name)
        if not found: raise ValueError('Unexpected coordinate name')
        tiles.setdefault(found[1],[]).append((name,int(found[2]),int(found[3])))
    with zipfile.ZipFile(archive_path) as archive:
        members={}
        for name in archive.namelist():
            found=re.search(r'top_potsdam_(\d+_\d+)_IRRG\.tif$',name)
            if found:
                if found[1] in members: raise ValueError('Duplicate TIFF tile')
                members[found[1]]=name
        for tile,rows in sorted(tiles.items()):
            with archive.open(members[tile]) as stream,Image.open(stream) as image:
                if image.mode!='RGB' or image.size!=(6000,6000): raise ValueError('Unexpected IRRG format')
                geo={'tiepoint':list(image.tag_v2[33922]),'pixel_scale':list(image.tag_v2[33550])}
                image.load()
                agreements=[]
                for name,x,y in rows:
                    patch=image.crop((x,y,x+512,y+512)); original=rgb/name
                    with Image.open(original) as reference:
                        if reference.size!=patch.size or reference.mode!='RGB': raise ValueError('Paired RGB geometry differs')
                        # Record native plane correspondence; do not infer wavelengths from PIL RGB names.
                        a=np.asarray(patch); b=np.asarray(reference)
                        agreements.append([[bool(np.array_equal(a[:,:,i],b[:,:,j])) for j in range(3)] for i in range(3)])
                    destination=output/name; destination.parent.mkdir(parents=True,exist_ok=True); patch.save(destination)
                    metadata['files_sha256'][name]=sha(destination); metadata['rgb_reference_sha256'][name]=sha(original)
                    if name.startswith('target_eval/'):
                        label=name.replace('/images/','/labels/'); destination=output/label
                        destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes((rgb/label).read_bytes())
                        metadata['files_sha256'][label]=sha(destination)
                metadata['tiles'][tile]={'patches':len(rows),**geo,'native_plane_exact_correspondence':agreements[0],
                    'correspondence_constant_across_selected_patches':all(v==agreements[0] for v in agreements)}
                print('IRRG exported',tile,len(rows),flush=True)
    metadata['preparer_sha256']=sha(__file__)
    (output/'METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(REPORT/'VH2PD_IRRG_PILOT.zip','w',zipfile.ZIP_STORED) as archive:
        for name in sorted(metadata['files_sha256']): archive.write(output/name,name)
        archive.write(output/'METADATA.json','METADATA.json')
    print('PASS: genuine IRRG pilot208images; eval labels copied without decoding; no target-train labels',flush=True)


if __name__=='__main__': main()
