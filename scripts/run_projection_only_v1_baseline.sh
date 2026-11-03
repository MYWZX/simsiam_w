#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}" scripts/run_projection_only_v1_baseline.py "$@"
