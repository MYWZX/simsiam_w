#!/usr/bin/env bash
# Matched dual-quality residual control; pretrained alpha is transferred.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: bash %s {pretrain|agiqa1k|agiqa3k|aigciqa2023} [--dry-run]\n' "$0" >&2
  exit 2
fi
stage="$1"; dry_run="${2:-}"
if [[ -n "$dry_run" && "$dry_run" != --dry-run ]]; then exit 2; fi
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
pretrain_name=richhf18k_cliplarge_dualquality_residual_layer3_seed42
checkpoint="outputs/$pretrain_name/best_component_anchor_pretrain.pth"
common=(--config "configs/dual_quality_residual/${stage}.yaml" --clip-large --seed 42 --text-context-tokens 4 --local-feature-stage layer3)
if [[ "$stage" == pretrain ]]; then
  run_name="$pretrain_name"
  cmd=(env HF_HUB_OFFLINE=1 "$python_bin" train_cm_pretrain.py "${common[@]}")
else
  selection_flags=(); selection_tag=""
  case "$stage" in
    agiqa1k) ;;
    agiqa3k|aigciqa2023) selection_flags=(--compare-selection-policies); selection_tag=_dualselect;;
    *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2;;
  esac
  run_name="${stage}_quality_cliplarge_layer3_dualquality_residual${selection_tag}_seed42"
  cmd=(env HF_HUB_OFFLINE=1 "$python_bin" train_cm_iqa.py "${common[@]}"
    --quality-global-feature g --align-global-feature g
    --quality-text-blend 0 --quality-text-lambda 0.005
    --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1
    --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients
    "${selection_flags[@]}" --pretrained-checkpoint "$checkpoint")
fi
cmd+=(--output-dir "outputs/$run_name" --results-csv "outputs/all_results_${run_name}.csv")
if [[ "$dry_run" == --dry-run ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
if [[ "$stage" != pretrain && ! -f "$checkpoint" ]]; then printf 'Missing checkpoint: %s\n' "$checkpoint" >&2; exit 1; fi
if [[ -e "outputs/$run_name" || -e "outputs/all_results_${run_name}.csv" ]]; then
  printf 'Output already exists for %s; preserve/rename it before a fresh run.\n' "$run_name" >&2
  exit 1
fi
exec "${cmd[@]}"
