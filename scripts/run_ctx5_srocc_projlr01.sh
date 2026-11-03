#!/usr/bin/env bash
# Same source, split and selector as the best baseline; only PI/PT LR is reduced.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${ABLATION_PYTHON:-/home/sll/anaconda3/envs/WZX-AI/bin/python}"
selection="${1:-all}"
if [[ $# -gt 1 ]]; then exit 2; fi
case "$selection" in
 pack) exec "$python_bin" scripts/pack_ctx5_srocc_projlr01_results.py;;
 all) datasets=(agiqa1k agiqa3k aigciqa2023);;
 agiqa1k|agiqa3k|aigciqa2023) datasets=("$selection");;
 *) echo 'Expected all|agiqa1k|agiqa3k|aigciqa2023|pack' >&2; exit 2;;
esac
# Reuse the existing component-anchor-selected source; no new pretraining.
env HF_HUB_OFFLINE=1 "$python_bin" - <<'PYVERIFY'
from pathlib import Path
import csv,json
root=Path('outputs/richhf18k_globalpi_ctx5_noce_noneg_dev_seed42')
source=root/'best_component_anchor_pretrain.pth'
if not source.is_file():raise FileNotFoundError(source)
csv.field_size_limit(20_000_000)
with root.with_suffix('.csv').open(encoding='utf-8-sig',newline='') as stream:
 finals=[json.loads(r['metrics_json']) for r in csv.DictReader(stream) if r['event']=='final']
if len(finals)!=1 or finals[0]['selection_metric']!='val_component_anchor_SROCC':
 raise ValueError('Expected completed original component-anchor-selected pretraining')
print('Reuse component-anchor source epoch:',finals[0]['best_component_anchor_epoch'])
PYVERIFY
for ds in "${datasets[@]}"; do
 env HF_HUB_OFFLINE=1 "$python_bin" - "$ds" <<'PYBASELINE'
import sys,yaml
from pathlib import Path
from scripts.run_unmerged_ii_experiments import verify_completed
cfg=yaml.safe_load(Path(f'configs/global_pi_ctx5_noce_noneg/{sys.argv[1]}.yaml').read_text())
verify_completed(cfg,'train_cm_iqa.py')
PYBASELINE
 prefix="$ds"
 if [[ "$ds" == agiqa1k ]]; then
  env HF_HUB_OFFLINE=1 "$python_bin" scripts/prepare_agiqa1k_subset1078.py
  prefix=agiqa1k_subset1078
 fi
 name="${prefix}_globalpi_ctx5_srocc_projlr01_noce_noneg_dev_seed42"
 if [[ -e "outputs/$name" || -e "outputs/$name.csv" ]]; then
  echo "Preserve existing output: outputs/$name" >&2; exit 1
 fi
 env HF_HUB_OFFLINE=1 "$python_bin" train_cm_iqa.py --config "configs/global_pi_ctx5_srocc_projlr01/$ds.yaml"
done
