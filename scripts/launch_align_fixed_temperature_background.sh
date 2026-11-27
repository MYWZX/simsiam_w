#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 0 ]]; then echo 'No arguments accepted' >&2; exit 2; fi
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
"$python_bin" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Two server GPUs required"'
# Full read-only preflight of source, controls and all existing outputs first.
"$python_bin" scripts/preflight_align_fixed_temperature.py
mkdir -p run_logs
log_dir=$(mktemp -d "run_logs/align_fixed_temperature_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
for queue in unit sharp; do
  gpu=0; [[ "$queue" == sharp ]] && gpu=1
  CUDA_VISIBLE_DEVICES="$gpu" nohup setsid bash scripts/run_align_fixed_temperature.sh "$queue" --continue > "$log_dir/$queue.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$log_dir/$queue.pid"
  printf 'GPU%s %s PID=%s log=%s/%s.log\n' "$gpu" "$queue" "$pid" "$log_dir" "$queue"
done
printf 'Monitor: tail -f %s/unit.log %s/sharp.log\n' "$log_dir" "$log_dir"
