# Quality pairwise ranking weight from the selected v2 tuning reference

This is the first new sweep based on configs/current_tuning_reference and
 docs/current_tuning_reference.json, NOT the original all-split42 baseline.
Fix train seed42, 1K subset1078 split3407 and 3K/2023 split42. Preserve Quality
Soft-SROCC coefficients0.03/0.02/0.04 for1K/3K/2023 respectively. In particular
2023 must use the completed qsrocc200 result as its control, never old0.02.

Change only train.lambda_rank_quality: original0.03, qrank050=0.015, qrank200=0.06.
This is downstream hard-pairwise Quality ranking. Margin0.02, hard_rank_ratio0.35,
Soft-SROCC tau0.02, Quality MSE coefficient1 and all Align loss coefficients remain
unchanged. No change to pretraining Component Anchors, Ordinal or other objectives.
The same two absolute Quality rank weights apply to all three datasets.

Reuse accepted epoch4 projection_only_v1 component-anchor-selected pretraining
source SHA256 fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
No pretraining/resume. Retain CLIP/ResNet initial LR2e-6, semantic_proj1e-5,
QA/PI/PT/quality1e-4, original scheduler/floor, inherited alpha, beta start0.50,
decay0.002, dropout0.20/default per-module values, original train/val/test,
60 epochs and maximum validation Quality SROCC selection. No merge or stacking
of earlier candidate LR/dropout/alpha/beta/decay changes.

Configs are derived from v2 and saved as launch_config.yaml under fresh v2 output
identifiers. Before any run, verify source/completion/config/mode/SHA and the
same-split control. The control CSV SHA and run_id must exactly match the selected
v2 record and its snapshot must match the selected v2 configuration. For2023,
reuse and validate the old run_quality_srocc qsrocc200 completion explicitly.
Wrong old0.02 control snapshots, changed CSVs or source mismatches are refused.

After training, validate all60 epoch records and that weighted Quality rank,
Soft-SROCC and MSE contributions equal their effective coefficients times raw
losses (relative tolerance1e-3, absolute1e-7 for AMP rounding). Actual optimizer
initial groups, training/split seeds and membership evidence are also checked.
No loss/trainer implementation is changed in this workflow; tests use the real
compute_losses to verify the objective change is rank-only.

Server setup and six-run sequence:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_quality_rank.sh all
```

all preflights every target path before launching and stops on any failure.
Existing directories/CSVs/archives are refused and partial outputs preserved.
After partial completion do not rerun all; use only unstarted individual commands:

```bash
bash scripts/run_quality_rank.sh qrank050 agiqa1k && bash scripts/run_quality_rank.sh qrank200 agiqa1k
bash scripts/run_quality_rank.sh qrank050 agiqa3k && bash scripts/run_quality_rank.sh qrank200 agiqa3k
bash scripts/run_quality_rank.sh qrank050 aigciqa2023 && bash scripts/run_quality_rank.sh qrank200 aigciqa2023
bash scripts/run_quality_rank.sh pack
```

After all six complete, upload
projection_only_v1_v2_quality_rank_1ksplit3407_results.tar.gz. Explicit allowlists
include selected v2 reference configs/JSON/Markdown, same-split control evidence
(including2023 Soft-SROCC0.04), candidate CSVs/configs/logs, runtime/optimizer/source/
launch/completion metadata and actual split manifests, excluding all weights.
Original controls are not rerun. Old runners retain their historical comparison
rules; do not use them as if they already adopted v2.

Default Python /home/sll/anaconda3/envs/WZX-AI/bin/python; ABLATION_PYTHON overrides.
HF_HUB_OFFLINE1 uses the verified CLIP cache. CPU full runs are refused. check-configs
validates six settings without server artifacts; all --dry-run verifies source/
controls and prints configs without training/writing. No GPU index is selected.

These remain single-seed exploratory evaluations on previously viewed splits.
Keep all outcomes and compare all four correlations against the exact v2 controls.
No new real-data training or server checkpoint loading ran in cloud. CPU tests
cover config isolation, actual rank-only objective changes and finite gradients,
old/wrong2023 control rejection, AMP-tolerant loss audits, six-run dispatch,
no-overwrite, CPU refusal and weight-free packaging with v2 evidence.
