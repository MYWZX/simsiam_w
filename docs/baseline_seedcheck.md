# Accepted baseline: downstream training-seed replication

Retain the accepted projection_only_v1 baseline and add train seeds 123 and 3407.
Seed42 already completed and is reused. This is a downstream-seed study with the
same seed42 pretraining checkpoint; it does NOT measure pretraining-seed variance
or establish end-to-end multi-seed robustness.

All configurations are copied exactly from configs/projection_only_v1_baseline
except top-level seed and output/run identifiers. split_seed remains 42 and all
original sample memberships are preserved (AGIQA-1K subset1078 train755/val108/test215).
No train+val merge. Sixty epochs, validation Quality SROCC maximum selection,
weight_decay=0.002, PI/PT LR=1e-4, global projection LR=1e-5, learnable inherited
QA alpha, architecture and losses remain unchanged. Neither unsuccessful QA
alpha-start candidates nor stronger-decay candidates are used as the baseline.

All six new runs use the same accepted epoch4 component-anchor-selected source:
SHA256 fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
The runner verifies source completion, cfg/mode/selection, checksum and the pinned
baseline record. It also checks the actual runtime seed and split_seed after each
completed run. No pretraining or resume is launched, and no existing output/CSV
or archive is overwritten. CPU full runs are refused. No GPU number is assigned.

Server commands, two independent seeds per dataset:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_baseline_seedcheck.sh seed123 agiqa1k && bash scripts/run_baseline_seedcheck.sh seed3407 agiqa1k
bash scripts/run_baseline_seedcheck.sh seed123 agiqa3k && bash scripts/run_baseline_seedcheck.sh seed3407 agiqa3k
bash scripts/run_baseline_seedcheck.sh seed123 aigciqa2023 && bash scripts/run_baseline_seedcheck.sh seed3407 aigciqa2023
bash scripts/run_baseline_seedcheck.sh pack
```

Each command explicitly names one training seed and dataset. Use check-configs to
validate six configs without source weights. --dry-run still verifies the actual
server source but invokes no trainer. Default Python is
/home/sll/anaconda3/envs/WZX-AI/bin/python; override via ABLATION_PYTHON. Offline
CLIP cache loading is the default, preserving the prior connectivity workaround.
The script does not impose new determinism/TF32/cuDNN settings; runtime metadata
records observed settings, and seed control alone is not bitwise reproducibility.

Packaging requires six complete new downstream runs and includes the accepted
seed42 baseline evidence. File: projection_only_v1_seedcheck_results.tar.gz.
Contents: CSVs, config snapshots, logs, runtime/launch/completion metadata and
source fingerprints; no model weights. Later report individual seed results and
mean plus sample standard deviation (ddof=1) across 42/123/3407, for all four
Quality/Align correlations. 1K Align is missing, not zero. Do not choose the best
seed as a replacement for the mean, change splits or retune parameters per seed.

Only CPU script/config tests were run in cloud; no new real-data training occurred.
