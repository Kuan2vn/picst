"""Create the self-contained Kaggle notebook using standard nbformat JSON."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def cell(kind, text):
        item = {"cell_type": kind, "id": f"gsd-{len(cells):03d}", "metadata": {},
                "source": text.strip().splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell("markdown", """
# GSD-SFUDA: frozen reliability diagnostics and integration smoke

Chọn **GPU T4** và bật **Internet** trong Kaggle. Code chỉ dùng cuda:0.
Notebook tải toàn bộ public Hugging Face repository, kiểm SHA-256, giải nén
code/data, chạy geometry tests, frozen-teacher diagnostics và smoke SND.

Đây là **development subset**, không phải bảng performance main:
16 ảnh target train không nhãn và 128 evaluation patches từ 14 Potsdam tiles.
Source checkpoint 8.500 bước là snapshot ban đầu; source_smoke chỉ có 2 bước.
HPC chạy source40k → warm20k → pilot4arms. Không trộn smoke với main results.
""")
    cell("code", """
from pathlib import Path
import subprocess, sys, os, json, hashlib, zipfile
REPO_ID = "kuan2/gsd-sfuda-remote-sensing"
ROOT = Path("/kaggle/working/gsd_sfuda")
DOWNLOAD = ROOT / "hub"
WORK = ROOT / "workspace"
RESULTS = Path("/kaggle/working/gsd_results")
RUN_INFERENCE = True
RUN_INTEGRATION_SMOKE = True
RUN_EXTENDED_CPU_SWEEP = False  # 21 lambda values; longer CPU work, no additional GPU forwards
for directory in (DOWNLOAD, WORK, RESULTS):
    directory.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
""")
    cell("code", """
from huggingface_hub import HfApi, snapshot_download
REVISION = HfApi(token=False).repo_info(REPO_ID, repo_type="dataset").sha
snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION,
                  local_dir=DOWNLOAD, token=False)  # download the entire repository
(RESULTS / "hub_revision.json").write_text(json.dumps({"repo": REPO_ID, "revision": REVISION}, indent=2))
manifest = json.loads((DOWNLOAD / "bundle_manifest.json").read_text())
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()
for item in manifest["files"]:
    path = DOWNLOAD / item["path"]
    assert path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], item["path"]
print("Verified", len(manifest["files"]), "files; revision", REVISION)
""")
    cell("code", """
def extract_verified(path, root):
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            destination = (root / item.filename).resolve()
            assert root.resolve() in destination.parents, item.filename
            assert ((item.external_attr >> 16) & 0o170000) != 0o120000, "Symlinks are not allowed"
        archive.extractall(root)
extract_verified(DOWNLOAD / "gsd_code_20261012.zip", WORK)
extract_verified(DOWNLOAD / "gsd_kaggle_irrg_v1.zip", WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(WORK / "requirements_gsd_kaggle.txt")], check=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Torch", torch.__version__, "device", DEVICE)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), "VRAM GiB", torch.cuda.get_device_properties(0).total_memory / 2**30)
    print("CUDA check:", torch.ones(2, device=DEVICE).sum().item())
DATA = WORK / "data/remote_sensing/gsd_kaggle_irrg_v1"
""")
    cell("markdown", """
Các kiểm tra sau xác nhận tọa độ và loss, không đo accuracy.
Đặc biệt q=1 phải khớp baseline loss/gradient và donor coefficients phải giữ nguyên.
""")
    cell("code", """
def run(arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=WORK, check=True)
run(["-m", "unittest", "discover", "-s", "tests", "-p", "test_gsd_reliability.py", "-v"])
""")
    cell("code", """
if RUN_INFERENCE:
    if DEVICE == "cpu":
        raise RuntimeError("Bật GPU T4 để chạy inference; hoặc đặt RUN_INFERENCE=False nếu chỉ phân tích cache đã có.")
    run(["-m", "scripts.diagnose_gsd_reliability",
         "--checkpoint", DOWNLOAD / "checkpoints/source_08500.pth",
         "--data-root", DATA, "--output", RESULTS / "source_08500",
         "--device", DEVICE, "--limit", "128",
         "--views", "ratio_1p2", "ratio_1p25", "ratio_1p8", "ratio_2"])
""")
    cell("code", """
import matplotlib.pyplot as plt
summary = json.loads((RESULTS / "source_08500/summary.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
curves = {"confidence": summary["confidence"]}
for view, record in summary["views"].items():
    curves[view] = record["lambda_sweep"]["1.0"]
for label, points in curves.items():
    for axis, key in zip(axes, ("micro_risk", "macro_predicted_class_risk")):
        axis.plot([100*p["coverage"] for p in points],
                  [100*p[key] for p in points], marker="o", label=label)
for axis, title in zip(axes, ("Pixel risk", "Macro risk over predicted classes")):
    axis.set(xlabel="Coverage (%)", ylabel="Selected label error (%)", title=title)
    axis.grid(alpha=.25)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / "risk_coverage.png", dpi=180)
plt.show()
print((RESULTS / "source_08500/REPORT.md").read_text())
""")
    cell("markdown", """
CPU sweep tái sử dụng scores, không chạy thêm teacher. Dùng để kiểm sensitivity;
main method vẫn khóa lambda=1, q=.5. Không dùng sweep này chọn tham số trên final test.
""")
    cell("code", """
if RUN_EXTENDED_CPU_SWEEP:
    import numpy as np
    from scripts.diagnose_gsd_reliability import analyze
    analyze(RESULTS / "source_08500/scores.npz", RESULTS / "cpu_lambda_sweep",
            lambdas=tuple(np.linspace(0, 2, 21)))
""")
    cell("code", """
if RUN_INTEGRATION_SMOKE:
    if DEVICE == "cpu":
        raise RuntimeError("Integration smoke cần GPU; đặt RUN_INTEGRATION_SMOKE=False nếu chỉ chạy CPU analysis.")
    run(["train_SND_remote.py", "smoke", "--protocol", "siamseg_sfuda_b2_v1",
         "--source-checkpoint", DOWNLOAD / "checkpoints/source_smoke.pth",
         "--data-root", DATA, "--gsd-config", WORK / "configs/gsd_sfuda/gsd_irrg.json",
         "--output", RESULTS / "integration_smoke", "--device", DEVICE,
         "--smoke-size", "512", "--smoke-batch-size", "1",
         "--eval-period", "0", "--workers", "0", "--seed", "1"])
""")
    cell("code", """
report_zip = Path("/kaggle/working/gsd_results_reports.zip")
with zipfile.ZipFile(report_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in RESULTS.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".md", ".png", ".log"):
            archive.write(path, path.relative_to(RESULTS))
print("Report download:", report_zip)
print("Reusable score cache:", RESULTS / "source_08500/scores.npz")
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}, "cells": cells}
    output = ROOT / "notebooks/gsd_sfuda_kaggle.ipynb"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in cells:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), str(output), "exec")
    print(f"Notebook validated: {len(cells)} cells -> {output}")


if __name__ == "__main__":
    main()
