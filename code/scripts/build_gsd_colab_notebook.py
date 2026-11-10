"""Build the independent CPU notebook and verified Drive input archive."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / "reports/gsd_sfuda_implementation_20261012"
    output = ROOT / "artifacts/gsd_colab_20261012"
    output.mkdir(parents=True, exist_ok=True)
    files = {"scores.npz": reports / "scores_source_08500_32.npz",
             "metadata.json": reports / "metadata.json",
             "gsd_tile_bootstrap.py": ROOT / "scripts/gsd_tile_bootstrap.py"}
    inventory = []
    with zipfile.ZipFile(output / "source_08500_32_inputs.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for name, path in files.items():
            content = path.read_bytes()
            archive.writestr(name, content)
            inventory.append({"path": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
        archive.writestr("input_manifest.json", json.dumps(inventory, indent=2) + "\n")
    cells = []
    def cell(kind, text):
        item = {"cell_type": kind, "id": f"colab-{len(cells):03d}", "metadata": {},
                "source": text.strip().splitlines(keepends=True)}
        if kind == "code":
            compile(text.strip(), "colab_cell", "exec")
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell("markdown", """
# GSD-SFUDA: tile bootstrap và sensitivity từ cache

Chạy trên **CPU runtime**, không cần GPU/compute units. Chọn Run all và cấp quyền
Drive của bạn khi được hỏi. Input đã đặt trong MyDrive/GSD-SFUDA_20261012.
Kết quả tự lưu vào cùng thư mục Drive; không chạy lại source/warm/SND.

Cache hiện tại: source8.500, 32 patches từ14 parent tiles, views1.8/2.0.
Đây là development preview; không phải kết quả full adaptation.
Bootstrap giữ các cutoffs từ cohort gốc, không refit ranking khi resample tile.
Không chọn main hyperparameters bằng sensitivity trên evaluation labels.
""")
    cell("code", """
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import sys, os, subprocess, json, hashlib, zipfile
from datetime import datetime, timezone
DRIVE_ROOT = Path('/content/drive/MyDrive/GSD-SFUDA_20261012')
INPUT_ZIP = DRIVE_ROOT / 'source_08500_32_inputs.zip'
LOCAL = Path('/content/gsd_sfuda_cpu')
LOCAL.mkdir(exist_ok=True)
STAMP = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
RESULTS = DRIVE_ROOT / 'colab_results' / STAMP
RESULTS.mkdir(parents=True, exist_ok=True)
RUN_SENSITIVITY = True
assert INPUT_ZIP.is_file(), f'Input cache missing: {INPUT_ZIP}'
with zipfile.ZipFile(INPUT_ZIP) as archive:
    inventory = json.loads(archive.read('input_manifest.json'))
    for item in inventory:
        dest = (LOCAL / item['path']).resolve()
        assert LOCAL.resolve() in dest.parents
        content = archive.read(item['path'])
        assert len(content) == item['bytes'] and hashlib.sha256(content).hexdigest() == item['sha256']
        dest.write_bytes(content)
sys.path.insert(0, str(LOCAL))
os.environ['OMP_NUM_THREADS'] = '2'
print('Verified inputs; results:', RESULTS)
""")
    cell("code", """
from gsd_tile_bootstrap import analyze
base = analyze(LOCAL / 'scores.npz', RESULTS / 'q0p5_lambda1', repeats=2000, seed=42)
for name, result in base['selectors'].items():
    lo, hi = result['micro_ci95']
    print(f'{name}: risk={100*result["micro_risk"]:.3f}%, tile CI=[{100*lo:.3f}, {100*hi:.3f}]%')
delta = base['paired_differences']['ratio_1p8_minus_ratio_2']
print('GSD minus generic (negative favors GSD):', delta)
""")
    cell("code", """
if RUN_SENSITIVITY:
    for q in (.25, .5, .75):
        for penalty in (.5, 1., 2.):
            if q == .5 and penalty == 1.:
                continue
            analyze(LOCAL / 'scores.npz', RESULTS / f'q{q}_lambda{penalty}',
                    coverage=q, penalty=penalty, repeats=2000, seed=42)
""")
    cell("code", """
import matplotlib.pyplot as plt
names = list(base['selectors'])
risk = [100*base['selectors'][n]['micro_risk'] for n in names]
lower = [risk[i] - 100*base['selectors'][n]['micro_ci95'][0] for i,n in enumerate(names)]
upper = [100*base['selectors'][n]['micro_ci95'][1] - risk[i] for i,n in enumerate(names)]
# Percentile CI can exclude the original point estimate; plot endpoints directly.
fig, ax = plt.subplots(figsize=(7, 4))
for i,n in enumerate(names):
    lo, hi = base['selectors'][n]['micro_ci95']
    ax.plot([i,i], [100*lo,100*hi], color='black')
ax.scatter(range(len(names)), risk)
ax.set(xticks=range(len(names)), xticklabels=names, ylabel='Selected pixel error (%)',
       title='Development tile bootstrap; fixed original cutoffs')
ax.grid(axis='y', alpha=.25)
fig.tight_layout()
fig.savefig(RESULTS / 'tile_bootstrap.png', dpi=180)
plt.show()
print('Saved to personal Drive:', RESULTS)
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"colab": {"name": "gsd_sfuda_colab_cpu.ipynb"},
                             "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                             "language_info": {"name": "python"}}}
    destination = ROOT / "notebooks/gsd_sfuda_colab_cpu.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(destination), "cells": len(cells),
                      "inputs": str(output / "source_08500_32_inputs.zip")}))


if __name__ == "__main__":
    main()
