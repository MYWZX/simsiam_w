#!/usr/bin/env bash
# Same matched layer3 residual method; only use the saved joint pretrain source.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: bash %s {agiqa1k|agiqa3k|aigciqa2023} [--dry-run]\n' "$0" >&2
  exit 2
fi
dataset="$1"; dry_run="${2:-}"
if [[ -n "$dry_run" && "$dry_run" != --dry-run ]]; then exit 2; fi
selection_flags=(); selection_tag=""
case "$dataset" in
  agiqa1k) ;;
  agiqa3k|aigciqa2023) selection_flags=(--compare-selection-policies); selection_tag=_dualselect;;
  *) printf 'Unknown dataset: %s\n' "$dataset" >&2; exit 2;;
esac
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
checkpoint=outputs/richhf18k_cliplarge_dualquality_residual_layer3_seed42/best_joint_pretrain.pth
run_name="${dataset}_quality_cliplarge_layer3_dualquality_residual${selection_tag}_pretrainjoint_seed42"
cmd=(env HF_HUB_OFFLINE=1 "$python_bin" train_cm_iqa.py
  --config "configs/dual_quality_residual_joint/${dataset}.yaml"
  --clip-large --seed 42 --text-context-tokens 4 --local-feature-stage layer3
  --quality-global-feature g --align-global-feature g
  --quality-text-blend 0 --quality-text-lambda 0.005
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients
  "${selection_flags[@]}" --pretrained-checkpoint "$checkpoint"
  --output-dir "outputs/$run_name" --results-csv "outputs/all_results_${run_name}.csv")
if [[ "$dry_run" == --dry-run ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
if [[ ! -f "$checkpoint" ]]; then
  printf 'Missing saved joint checkpoint: %s\n' "$checkpoint" >&2
  exit 1
fi
if [[ -e "outputs/$run_name" || -e "outputs/all_results_${run_name}.csv" ]]; then
  printf 'Output already exists for %s; preserve/rename it before a fresh run.\n' "$run_name" >&2
  exit 1
fi
exec "${cmd[@]}"
