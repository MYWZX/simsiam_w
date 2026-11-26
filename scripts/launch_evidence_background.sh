#!/usr/bin/env bash
# Launch only when the user executes on the GPU server, never invoked by tests.
set -euo pipefail
cd "$(dirname "$0")/.."
recover=false
if [[ "${1:-}" == --recover-incomplete ]]; then recover=true; shift; fi
if [[ $# -ne 0 ]]; then echo 'Only --recover-incomplete is accepted' >&2; exit 2; fi
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
"$python_bin" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Two server GPUs required"'
"$python_bin" scripts/run_align_evidence.py legacy check-configs
"$python_bin" scripts/run_align_evidence.py kurt6 check-configs
if "$recover"; then
  # Prove all sources and states before preserving/restarting downstream files.
  "$python_bin" scripts/run_nss_kurtosis.py recover-pretrain --dry-run
  "$python_bin" scripts/recover_align_evidence.py legacy --execute
fi
mkdir -p run_logs
log_dir=$(mktemp -d "run_logs/evidence_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
for arm in legacy kurt6; do
  gpu=0; [[ "$arm" == kurt6 ]] && gpu=1
  extra=()
  if "$recover"; then extra=(--recover); fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup setsid bash scripts/run_evidence_chain.sh "$arm" "${extra[@]}" > "$log_dir/$arm.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$log_dir/$arm.pid"
  printf 'GPU%s %s PID=%s log=%s/%s.log\n' "$gpu" "$arm" "$pid" "$log_dir" "$arm"
done
printf 'Monitor: tail -f %s/legacy.log %s/kurt6.log\n' "$log_dir" "$log_dir"
