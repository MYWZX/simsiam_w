#!/usr/bin/env bash
# A fresh full run, or downstream stages after the new pretraining completes.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:-all}"
case "$stage" in
 all) bash scripts/run_protocol_v2.sh pretrain both_noce_noneg;;
 downstream) ;;
 *) echo "Usage: bash $0 [all|downstream]" >&2; exit 2;;
esac
for dataset in agiqa1k agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh dev both_noce_noneg "$dataset"
  bash scripts/run_protocol_v2.sh plan both_noce_noneg "$dataset"
  bash scripts/run_protocol_v2.sh refit both_noce_noneg "$dataset"
done
