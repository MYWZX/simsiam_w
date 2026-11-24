#!/usr/bin/env bash
# Preserve incomplete phases, then detach the two independent GPU queues.
set -euo pipefail
cd "$(dirname "$0")/.."
protected_python="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
command -v nohup >/dev/null
command -v setsid >/dev/null
"$protected_python" scripts/recover_protected_local.py --execute
mkdir -p run_logs
protected_log_dir=$(mktemp -d "run_logs/protected_restart_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
CUDA_VISIBLE_DEVICES="${PROTECTED_DIRECT_GPU:-0}" nohup setsid bash scripts/run_protected_local.sh direct --continue \
  > "$protected_log_dir/direct.log" 2>&1 < /dev/null &
protected_direct_pid=$!
CUDA_VISIBLE_DEVICES="${PROTECTED_WARM_GPU:-1}" nohup setsid bash scripts/run_protected_local.sh warm-chain --continue \
  > "$protected_log_dir/warm_chain.log" 2>&1 < /dev/null &
protected_warm_pid=$!
printf '%s\n' "$protected_direct_pid" > "$protected_log_dir/direct.pid"
printf '%s\n' "$protected_warm_pid" > "$protected_log_dir/warm_chain.pid"
printf 'GPU0 direct PID=%s | log=%s/direct.log\n' "$protected_direct_pid" "$protected_log_dir"
printf 'GPU1 warm-chain PID=%s | log=%s/warm_chain.log\n' "$protected_warm_pid" "$protected_log_dir"
printf 'Monitor: tail -f %q/direct.log %q/warm_chain.log\n' "$protected_log_dir" "$protected_log_dir"
