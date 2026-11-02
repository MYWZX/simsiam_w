#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
stage="${1:-all}"
selection="${2:-all}"
if [[ $# -gt 2 ]]; then exit 2; fi
case "$stage" in all|pretrain|downstream|refit) ;; *) echo 'Expected all|pretrain|downstream|refit' >&2; exit 2;; esac
case "$selection" in all) datasets=(agiqa1k agiqa3k aigciqa2023);; agiqa1k|agiqa3k|aigciqa2023) datasets=("$selection");; *) exit 2;; esac
if [[ "$stage" == pretrain && "$selection" != all ]]; then exit 2; fi
fresh() {
 local output="$1"; shift
 if [[ -e "$output" || -e "$output.csv" ]]; then echo "Output exists: $output; preserved." >&2; exit 1; fi
 env HF_HUB_OFFLINE=1 "$python_bin" "$@"
}
if [[ "$stage" == all || "$stage" == pretrain ]]; then
 fresh outputs/richhf18k_globalpi_ctx5_loss_noce_noneg_dev_seed42 train_cm_pretrain.py --config configs/global_pi_ctx5_loss_noce_noneg/pretrain.yaml
fi
if [[ "$stage" == pretrain ]]; then exit 0; fi
env HF_HUB_OFFLINE=1 "$python_bin" - <<'PYVERIFY'
from pathlib import Path
import csv, json, yaml
root=Path('outputs/richhf18k_globalpi_ctx5_loss_noce_noneg_dev_seed42')
checkpoint=root/'best_pretrain.pth'
if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
with root.with_suffix('.csv').open(encoding='utf-8-sig',newline='') as stream:
 rows=list(csv.DictReader(stream))
finals=[json.loads(r['metrics_json']) for r in rows if r['event']=='final']
if len(finals)!=1 or finals[0]['selection_metric']!='val_loss':
 raise ValueError('Expected one completed loss-selected pretraining run')
cfg=yaml.safe_load((root/'config_snapshot.yaml').read_text())
if (cfg['train']['pretrain_selection_mode']!='min' or
    cfg['model']['text_encoder']['learnable_context_tokens']!=5):
 raise ValueError('Expected five-context, minimum-loss pretrained source')
PYVERIFY
for ds in "${datasets[@]}"; do
 name="${ds}_globalpi_ctx5_loss_noce_noneg_dev_seed42"
 if [[ "$ds" == agiqa1k ]]; then
  env HF_HUB_OFFLINE=1 "$python_bin" scripts/prepare_agiqa1k_subset1078.py
  name=agiqa1k_subset1078_globalpi_ctx5_loss_noce_noneg_dev_seed42
 fi
 if [[ "$stage" != refit ]]; then
  fresh "outputs/$name" train_cm_iqa.py --config "configs/global_pi_ctx5_loss_noce_noneg/$ds.yaml"
  env HF_HUB_OFFLINE=1 "$python_bin" scripts/prepare_protocol_refit.py --snapshot "outputs/$name/config_snapshot.yaml" --csv "outputs/$name.csv" --prefix "outputs/globalpi_ctx5_loss_plans/$ds"
 fi
 fresh "outputs/${name/_dev_/_refit80_}" train_cm_iqa.py --config "outputs/globalpi_ctx5_loss_plans/$ds.yaml" --refit-plan "outputs/globalpi_ctx5_loss_plans/$ds.json"
done
