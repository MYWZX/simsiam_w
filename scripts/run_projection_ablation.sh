#!/usr/bin/env bash
# Three independent pretrain/downstream ablations against the g/global baseline.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: bash %s {noce|mos_pq|shared_pi_pt} {pretrain|agiqa1k|agiqa3k|aigciqa2023} [--dry-run]\n' "$0" >&2
  exit 2
fi
mode="$1"; stage="$2"; dry_run="${3:-}"
case "$mode" in noce|mos_pq|shared_pi_pt) ;; *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2;; esac
if [[ -n "$dry_run" && "$dry_run" != --dry-run ]]; then exit 2; fi
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
pretrain_name="richhf18k_cliplarge_${mode}_seed42"
checkpoint="outputs/${pretrain_name}/best_component_anchor_pretrain.pth"
common=(--clip-large --seed 42 --text-context-tokens 4)
if [[ "$stage" == pretrain ]]; then
  run_name="$pretrain_name"
  cmd=(env HF_HUB_OFFLINE=1 "$python_bin" train_cm_pretrain.py --config "configs/projection_ablations/pretrain_${mode}.yaml" "${common[@]}")
else
  selection_flags=(); selection_tag=""
  case "$stage" in
    agiqa1k) ;;
    agiqa3k|aigciqa2023) selection_flags=(--compare-selection-policies); selection_tag=_dualselect;;
    *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2;;
  esac
  run_name="${stage}_quality_cliplarge_layer3_safeinput_noneg${selection_tag}_qualityaligng_${mode}_seed42"
  bank_flags=()
  if [[ "$mode" != noce ]]; then bank_flags=(--quality-text-blend 0 --quality-text-lambda 0.005); fi
  cmd=(env HF_HUB_OFFLINE=1 "$python_bin" train_cm_iqa.py --config "configs/projection_ablations/${stage}_${mode}.yaml"
    "${common[@]}" --local-feature-stage layer3 --reinit-local-projection
    --quality-global-feature g --align-global-feature g "${bank_flags[@]}"
    --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1
    --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients
    "${selection_flags[@]}" --pretrained-checkpoint "$checkpoint")
fi
cmd+=(--output-dir "outputs/$run_name" --results-csv "outputs/all_results_${run_name}.csv")
if [[ "$dry_run" == --dry-run ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
if [[ "$stage" != pretrain && ! -f "$checkpoint" ]]; then printf 'Missing checkpoint: %s\n' "$checkpoint" >&2; exit 1; fi
# A rerun must not append unrelated CSV records or overwrite provenance.
if [[ -e "outputs/all_results_${run_name}.csv" || -e "outputs/$run_name" ]]; then
  printf 'Output already exists for %s; preserve or rename it before starting a fresh run.\n' "$run_name" >&2
  exit 1
fi
exec "${cmd[@]}"
