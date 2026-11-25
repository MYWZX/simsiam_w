#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
fair_python="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
command -v nohup >/dev/null
command -v setsid >/dev/null
"$fair_python" scripts/run_align_40plus20.py check-inputs
mkdir -p run_logs
fair_log_dir=$(mktemp -d "run_logs/align4020_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
CUDA_VISIBLE_DEVICES=0 nohup setsid bash scripts/run_align_40plus20.sh a40local20 agiqa3k --continue \
  > "$fair_log_dir/agiqa3k.log" 2>&1 < /dev/null &
fair_1_pid=$!
CUDA_VISIBLE_DEVICES=1 nohup setsid bash scripts/run_align_40plus20.sh a40local20 aigciqa2023 --continue \
  > "$fair_log_dir/aigciqa2023.log" 2>&1 < /dev/null &
fair_2_pid=$!
printf '%s\n' "$fair_1_pid" > "$fair_log_dir/agiqa3k.pid"
printf '%s\n' "$fair_2_pid" > "$fair_log_dir/aigciqa2023.pid"
printf 'GPU0 AGIQA3K PID=%s | GPU1 AIGCIQA2023 PID=%s\n' "$fair_1_pid" "$fair_2_pid"
printf 'Monitor: tail -f %q/agiqa3k.log %q/aigciqa2023.log\n' "$fair_log_dir" "$fair_log_dir"
