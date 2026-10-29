#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
exec env HF_HUB_OFFLINE=1 "$python_bin" scripts/run_unmerged_ii_experiments.py "$@"
