#!/usr/bin/env bash
# Explicit server-only ordered queue. --continue skips only verified COMPLETED runs.
set -euo pipefail
cd "$(dirname "$0")/.."
arm="${1:?Specify legacy or kurt6}"
case "$arm" in legacy|kurt6) ;; *) echo 'Expected legacy or kurt6' >&2; exit 2;; esac
shift
if [[ $# -ne 0 ]]; then echo 'No extra arguments accepted' >&2; exit 2; fi
mkdir -p run_logs
exec 9>"run_logs/evidence_${arm}.lock"
flock -n 9 || { echo "Evidence queue already running: $arm" >&2; exit 1; }
if [[ "$arm" == kurt6 ]]; then
  bash scripts/run_nss_kurtosis.sh continue
fi
bash scripts/run_align_evidence.sh "$arm" all --continue
