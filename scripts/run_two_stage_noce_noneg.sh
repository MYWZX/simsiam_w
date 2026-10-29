#!/usr/bin/env bash
# A fresh full run, or downstream stages after the new pretraining completes.
set -euo pipefail
cd "$(dirname "$0")/.."
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
case "$stage" in
 all) bash scripts/run_protocol_v2.sh pretrain both_noce_noneg;;
 downstream) ;;
 *) echo "Usage: bash $0 [all|downstream] [all|agiqa1k|agiqa3k|aigciqa2023]" >&2; exit 2;;
esac
for dataset in "${datasets[@]}"; do
  bash scripts/run_protocol_v2.sh dev both_noce_noneg "$dataset"
  bash scripts/run_protocol_v2.sh plan both_noce_noneg "$dataset"
  bash scripts/run_protocol_v2.sh refit both_noce_noneg "$dataset"
done
