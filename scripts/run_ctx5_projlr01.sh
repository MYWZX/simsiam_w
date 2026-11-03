#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
dataset="${1:-all}"
phase="${2:-all}"
if [[ $# -gt 2 ]]; then exit 2; fi
if [[ "$dataset" == pack ]]; then
 exec "$python_bin" scripts/pack_ctx5_projlr01_results.py
fi
case "$dataset" in all) datasets=(agiqa1k agiqa3k aigciqa2023);; agiqa1k|agiqa3k|aigciqa2023) datasets=("$dataset");; *) exit 2;; esac
case "$phase" in all|dev|plan|refit) ;; *) exit 2;; esac
fresh() {
 local output="$1"; shift
 if [[ -e "$output" || -e "$output.csv" ]]; then echo "Preserve existing output: $output" >&2; exit 1; fi
 env HF_HUB_OFFLINE=1 "$python_bin" "$@"
}
env HF_HUB_OFFLINE=1 "$python_bin" scripts/prepare_ctx5_loss_transfer.py --trust-local-training-checkpoint
for ds in "${datasets[@]}"; do
 prefix="$ds"
 if [[ "$ds" == agiqa1k ]]; then
  env HF_HUB_OFFLINE=1 "$python_bin" scripts/prepare_agiqa1k_subset1078.py
  prefix=agiqa1k_subset1078
 fi
 name="${prefix}_globalpi_ctx5_loss_projlr01_noce_noneg_dev_seed42"
 plan="outputs/globalpi_ctx5_loss_projlr01_plans/$ds"
 if [[ "$phase" == all || "$phase" == dev ]]; then
  fresh "outputs/$name" train_cm_iqa.py --config "configs/global_pi_ctx5_loss_projlr01/$ds.yaml"
 fi
 if [[ "$phase" == all || "$phase" == plan ]]; then
  env HF_HUB_OFFLINE=1 "$python_bin" scripts/prepare_protocol_refit.py --snapshot "outputs/$name/config_snapshot.yaml" --csv "outputs/$name.csv" --prefix "$plan"
 fi
 if [[ "$phase" == all || "$phase" == refit ]]; then
  fresh "outputs/${name/_dev_/_refit80_}" train_cm_iqa.py --config "$plan.yaml" --refit-plan "$plan.json"
 fi
done
