#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec "${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}" scripts/run_local_lr.py "$@"
