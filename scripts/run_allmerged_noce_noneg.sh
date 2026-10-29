#!/usr/bin/env bash
# New independent experiment: merge train+val in BOTH training stages.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
stage="${1:-all}"
selection="${2:-all}"
if [[ $# -gt 2 ]]; then echo "Too many arguments" >&2; exit 2; fi
case "$selection" in
 all) datasets=(agiqa1k agiqa3k aigciqa2023);;
 agiqa1k|agiqa3k|aigciqa2023) datasets=("$selection");;
 *) echo "Unknown downstream dataset: $selection" >&2; exit 2;;
esac
if [[ "$stage" != downstream && "$selection" != all ]]; then
  echo "A dataset selector is supported only with downstream." >&2; exit 2
fi
run_fresh() {
  local output="$1"; shift
  if [[ -e "$output" || -e "$output.csv" ]]; then
    echo "Output exists: $output; continue from the unfinished stage, without overwriting." >&2
    exit 1
  fi
  env HF_HUB_OFFLINE=1 "$python_bin" "$@"
}
case "$stage" in all|pretrain-dev|pretrain-refit|downstream) ;; *) echo "Usage: bash $0 [all|pretrain-dev|pretrain-refit|downstream] [all|agiqa1k|agiqa3k|aigciqa2023]" >&2; exit 2;; esac
if [[ "$stage" == all || "$stage" == pretrain-dev ]]; then
  run_fresh outputs/richhf18k_allmerged_noce_noneg_pretrain_dev_seed42 \
    train_cm_pretrain.py --config configs/allmerged_noce_noneg/pretrain_dev.yaml
fi
if [[ "$stage" == all || "$stage" == pretrain-refit ]]; then
  run_fresh outputs/richhf18k_allmerged_noce_noneg_refit_seed42 \
    scripts/refit_merged_pretrain.py \
    --snapshot outputs/richhf18k_allmerged_noce_noneg_pretrain_dev_seed42/config_snapshot.yaml \
    --csv outputs/richhf18k_allmerged_noce_noneg_pretrain_dev_seed42.csv \
    --output-dir outputs/richhf18k_allmerged_noce_noneg_refit_seed42
fi
if [[ "$stage" == all || "$stage" == downstream ]]; then
  for dataset in "${datasets[@]}"; do
    name="${dataset}_allmerged_noce_noneg_dev_seed42"
    prefix="outputs/allmerged_noce_noneg_plans/$dataset"
    run_fresh "outputs/$name" train_cm_iqa.py --config "configs/allmerged_noce_noneg/$dataset.yaml"
    "$python_bin" scripts/prepare_protocol_refit.py --snapshot "outputs/$name/config_snapshot.yaml" \
      --csv "outputs/$name.csv" --prefix "$prefix"
    run_fresh "outputs/${dataset}_allmerged_noce_noneg_refit80_seed42" \
      train_cm_iqa.py --config "$prefix.yaml" --refit-plan "$prefix.json"
  done
fi
