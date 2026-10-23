#!/usr/bin/env bash
# Two isolated downstream controls against the retained layer4->layer3 baseline.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: bash %s {g_local|z_global} {agiqa1k|agiqa3k|aigciqa2023} [--dry-run]\n' "$0" >&2
  exit 2
fi
mode="$1"
dataset="$2"
dry_run="${3:-}"
if [[ -n "$dry_run" && "$dry_run" != --dry-run ]]; then
  printf 'Unknown option: %s\n' "$dry_run" >&2
  exit 2
fi
case "$mode" in
  g_local) route_flags=(--quality-global-feature g --align-global-feature g); run_tag=qualityaligng ;;
  z_global) route_flags=(--quality-global-feature z --quality-no-local-branch); run_tag=zglobalonly ;;
  *) printf 'Unknown ablation: %s\n' "$mode" >&2; exit 2 ;;
esac
selection_flags=()
case "$dataset" in
  agiqa1k) selection_tag= ;;
  agiqa3k|aigciqa2023) selection_flags=(--compare-selection-policies); selection_tag=_dualselect ;;
  *) printf 'Unknown dataset: %s\n' "$dataset" >&2; exit 2 ;;
esac
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
checkpoint=outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth
run_name="${dataset}_quality_cliplarge_layer3_safeinput_noneg${selection_tag}_${run_tag}_seed42"
cmd=(env HF_HUB_OFFLINE=1 "$python_bin" train_cm_iqa.py
  --config "configs/${dataset}_cm_iqa_decoupled_quality_align.yaml"
  --clip-large --seed 42 --text-context-tokens 4
  --local-feature-stage layer3 --reinit-local-projection
  --quality-text-blend 0 --quality-text-lambda 0.005
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients
  "${route_flags[@]}" "${selection_flags[@]}"
  --pretrained-checkpoint "$checkpoint"
  --output-dir "outputs/$run_name"
  --results-csv "outputs/all_results_$run_name.csv")
if [[ "$dry_run" == --dry-run ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi
if [[ ! -f "$checkpoint" ]]; then
  printf 'Missing pretrained checkpoint: %s\n' "$checkpoint" >&2
  exit 1
fi
exec "${cmd[@]}"
