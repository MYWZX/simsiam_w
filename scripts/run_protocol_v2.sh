#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
stage="${1:-}"; mode="${2:-baseline}"; dataset="${3:-agiqa1k}"
case "$mode" in baseline|align_aux|proj_lr|qa_bn|resnet_bn|noce_noneg) ;; *) exit 2;; esac
case "$dataset" in agiqa1k|agiqa3k|aigciqa2023) ;; *) exit 2;; esac
if [[ "$mode" == align_aux && "$dataset" == agiqa1k ]]; then
 echo '1K has no Align loss; reuse baseline development/refit rather than duplicate training.' >&2; exit 2
fi
name="${dataset}_protocolv2_${mode}_dev_seed42"
prefix="outputs/protocolv2_plans/${dataset}_${mode}"
case "$stage" in
 pretrain) config=configs/protocol_v2/pretrain.yaml; output=outputs/richhf18k_protocolv2_shared_seed42; script=train_cm_pretrain.py;;
 dev) config="configs/protocol_v2/${dataset}_${mode}.yaml"; output="outputs/$name"; script=train_cm_iqa.py;;
 plan) exec "$python_bin" scripts/prepare_protocol_refit.py --snapshot "outputs/$name/config_snapshot.yaml" --csv "outputs/$name.csv" --prefix "$prefix";;
 refit) config="$prefix.yaml"; output="outputs/${dataset}_protocolv2_${mode}_refit80_seed42"; script=train_cm_iqa.py;;
 *) echo "Usage: bash $0 {pretrain|dev|plan|refit} [baseline|align_aux|proj_lr|qa_bn|resnet_bn|noce_noneg] [agiqa1k|agiqa3k|aigciqa2023]" >&2; exit 2;;
esac
if [[ -e "$output" || -e "$output.csv" ]]; then echo "Output exists: $output; preserve it before rerunning." >&2; exit 1; fi
extra=(); if [[ "$stage" == refit ]]; then extra=(--refit-plan "$prefix.json"); fi
exec env HF_HUB_OFFLINE=1 "$python_bin" "$script" --config "$config" "${extra[@]}"
