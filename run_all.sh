#!/usr/bin/env bash
# Run the pending baselines one after another (an interrupted run resumes; finished ones are skipped).
#   bash run_all.sh                               # default list below
#   JOBS="picst:irrg dtst:rgb" bash run_all.sh    # custom list (method:setting)
#   HOURS=10 bash run_all.sh                      # pause each run safely after ~10 h; re-run to resume
# Run it inside tmux on the GPU node. After every job it refreshes results/ and results.zip (send results.zip).
set -uo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
env_name=${CONDA_ENV:-picst}
jobs=${JOBS:-"s3t_sfda:irrg stcgn:irrg dtst:rgb stable_st:rgb s3t_sfda:rgb stcgn:rgb"}
hours=${HOURS:-1000000}

if ! command -v conda >/dev/null 2>&1; then
    for hook in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
        if [[ -f "$hook" ]]; then . "$hook"; break; fi
    done
fi
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "$env_name"; fi
cd "$root"

# experiment folder that run.py writes (see METHODS in run.py)
exp_name() {
    case "$1:$2" in
        dtst:*) echo pheno_null ;;
        picst:irrg) echo pheno_c0_c1plaus ;;
        picst:rgb) echo pheno_c0multi_c1plaus ;;
        *) echo "$1" ;;
    esac
}

for job in $jobs; do
    method=${job%%:*}
    setting=${job##*:}
    state="runs/$setting/$(exp_name "$method" "$setting")/seed1/RUN_STATE.json"
    if [[ -f "$state" ]] && grep -q '"status": "finished"' "$state"; then
        echo "== skip $job (finished)"
        continue
    fi
    echo "== $(date '+%F %T') start $job"
    python run.py --method "$method" --setting "$setting" --hours "$hours" || echo "!! $job failed (see $state)"
    python collect_results.py >/dev/null || true
done
python collect_results.py
