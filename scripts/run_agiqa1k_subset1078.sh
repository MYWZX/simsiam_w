#!/usr/bin/env bash
# Only downstream 1K; existing pretraining weights are reused.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
mode="${1:-all}"
phase="${2:-all}"
if [[ $# -gt 2 ]]; then echo "Too many arguments" >&2; exit 2; fi
case "$phase" in all|dev|plan|refit) ;; *) echo "Unknown phase: $phase" >&2; exit 2;; esac
case "$mode" in
 prepare) methods=();;
 both|allmerged) methods=("$mode");;
 all) methods=(both allmerged);;
 *) echo "Usage: bash $0 [prepare|both|allmerged|all] [all|dev|plan|refit]" >&2; exit 2;;
esac
"$python_bin" scripts/prepare_agiqa1k_subset1078.py
run_fresh() {
  local output="$1"; shift
  if [[ -e "$output" || -e "$output.csv" ]]; then
    echo "Output exists: $output; resume from the unfinished stage without overwriting." >&2; exit 1
  fi
  env HF_HUB_OFFLINE=1 "$python_bin" "$@"
}
for method in "${methods[@]}"; do
  if [[ "$method" == both ]]; then
    name=agiqa1k_subset1078_protocolv2_both_noce_noneg_dev_seed42
  else
    name=agiqa1k_subset1078_allmerged_noce_noneg_dev_seed42
  fi
  prefix="outputs/subset1078_plans/$method"
  if [[ "$phase" == all || "$phase" == dev ]]; then
    run_fresh "outputs/$name" train_cm_iqa.py --config "configs/agiqa1k_subset1078/$method.yaml"
  fi
  if [[ "$phase" == all || "$phase" == plan ]]; then
    "$python_bin" scripts/prepare_protocol_refit.py \
      --snapshot "outputs/$name/config_snapshot.yaml" --csv "outputs/$name.csv" --prefix "$prefix"
  fi
  if [[ "$phase" == all || "$phase" == refit ]]; then
    run_fresh "outputs/${name/_dev_/_refit80_}" train_cm_iqa.py \
      --config "$prefix.yaml" --refit-plan "$prefix.json"
  fi
done
