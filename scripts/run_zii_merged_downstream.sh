#!/usr/bin/env bash
# Refit only: reuse completed fused-z development results and pretraining.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
selection="${1:-all}"
phase="${2:-all}"
if [[ $# -gt 2 ]]; then echo 'Too many arguments' >&2; exit 2; fi
case "$selection" in
 all) datasets=(agiqa1k agiqa3k aigciqa2023);;
 agiqa1k|agiqa3k|aigciqa2023) datasets=("$selection");;
 *) echo 'Expected all|agiqa1k|agiqa3k|aigciqa2023' >&2; exit 2;;
esac
case "$phase" in all|plan|refit) ;; *) echo 'Expected all|plan|refit' >&2; exit 2;; esac
for dataset in "${datasets[@]}"; do
  config="configs/zii_noce_noneg/${dataset}_unmerged.yaml"
  name="${dataset}_zii_noce_noneg_unmerged_seed42"
  if [[ "$dataset" == agiqa1k ]]; then name="agiqa1k_subset1078_zii_noce_noneg_unmerged_seed42"; fi
  prefix="outputs/zii_refit_plans/$dataset"
  # Check the completed trajectory, config and actual source hash before reuse.
  env HF_HUB_OFFLINE=1 "$python_bin" - "$config" <<'PY'
import sys, yaml
from scripts.run_unmerged_ii_experiments import verify_completed
verify_completed(yaml.safe_load(open(sys.argv[1])), 'train_cm_iqa.py')
PY
  if [[ "$phase" == all || "$phase" == plan ]]; then
    env HF_HUB_OFFLINE=1 "$python_bin" scripts/prepare_protocol_refit.py \
      --snapshot "outputs/$name/config_snapshot.yaml" \
      --csv "outputs/$name.csv" --prefix "$prefix"
  fi
  if [[ "$phase" == all || "$phase" == refit ]]; then
    output="outputs/${name/_unmerged_/_refit80_}"
    if [[ -e "$output" || -e "$output.csv" ]]; then
      echo "Output exists: $output; preserved without overwriting." >&2; exit 1
    fi
    env HF_HUB_OFFLINE=1 "$python_bin" train_cm_iqa.py \
      --config "$prefix.yaml" --refit-plan "$prefix.json"
  fi
done
