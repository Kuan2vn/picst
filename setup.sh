#!/usr/bin/env bash
# One-time setup: conda env "picst" + requirements + data bundle.  Run: bash setup.sh
# Needs read access to the private HF dataset: put HF_TOKEN=hf_xxx in ./.env, or run `hf auth login` first.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
env_name=${CONDA_ENV:-picst}

if [[ -f "$root/.env" ]]; then set -a; . "$root/.env"; set +a; fi
if ! command -v conda >/dev/null 2>&1; then
    for hook in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
        if [[ -f "$hook" ]]; then . "$hook"; break; fi
    done
fi
command -v conda >/dev/null 2>&1 || { echo 'conda not found' >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | awk '{print $1}' | grep -qx "$env_name" || conda create -y -n "$env_name" -c conda-forge --override-channels python=3.10
conda activate "$env_name"

python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$root/requirements.txt"

cd "$root"
python download_data.py          # about 6.6 GB into ./data
for f in data/source_40000.pth data/gsd_sfuda_b2_irrg_v1 data/siamseg_sfuda_b2_v1 data/snd_warm; do
    [[ -e "$f" ]] || { echo "Missing after download: $f" >&2; exit 1; }
done
echo 'Setup complete. Next: bash run_all.sh'
