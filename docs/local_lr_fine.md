# Fine local backbone LR: six downstream runs (first phase only)

Implement the requested first phase: llr0125 multiplier0.0125, LR1.25e-6;
llr0150 multiplier0.015, LR1.5e-6. Each runs on agiqa1k, agiqa3k and aigciqa2023,
exactly six new downstream runs. The proposed later3e-6/5e-6 phase is not launched
or included in this workflow. Existing1e-6,2e-6,4e-6 runs are references, not reruns.

Fix the accepted projection_only_v1 epoch4 pretraining source, SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
Train seed42; 1K subset1078 split3407; 3K/2023 split42. Preserve CLIP2e-6, QA/PI/PT
1e-4, semantic_proj1e-5, independent P0 projection LR, inherited learnable alpha,
score beta start0.50, decay0.002, all model shapes/losses, train/val/test and
60 epochs with maximum validation Quality SROCC selection. No merge or pretraining.
Only local_encoder_mult changes; its local_encoder.proj stays with that group.
The independent local_patch_proj remains in quality, unchanged. Both new rates
follow the original cosine scheduler down to1e-6; no schedule-floor change.

Effective configs are derived deterministically and saved as launch_config.yaml
under unique fresh output directories. The script checks source completion/mode/
SHA, fixed config/seed/split and matching controls; records actual optimizer initial
groups and actual split indices. Package verification checks file hashes and
membership evidence wherever it exists. No silent hyperparameter overrides or
candidate-source substitution. Legacy runner behavior remains unchanged.

Server setup and the explicit six-run sequence:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_local_lr_fine.sh all
```

all preflights source/control and ALL six target paths before the first launch.
It executes1K llr0125/llr0150, then3K both, then2023 both. A failure stops immediately
and partial outputs remain. If partially completed, do not rerun all or delete
results: use the individual commands for only the unstarted remaining runs.
Existing directories/CSV/archive files are refused, never overwritten or resumed.

Individual commands, grouped by dataset:

```bash
bash scripts/run_local_lr_fine.sh llr0125 agiqa1k && bash scripts/run_local_lr_fine.sh llr0150 agiqa1k
bash scripts/run_local_lr_fine.sh llr0125 agiqa3k && bash scripts/run_local_lr_fine.sh llr0150 agiqa3k
bash scripts/run_local_lr_fine.sh llr0125 aigciqa2023 && bash scripts/run_local_lr_fine.sh llr0150 aigciqa2023
```

After all six complete:

```bash
bash scripts/run_local_lr_fine.sh pack
```

Upload projection_only_v1_local_lr_fine_1ksplit3407_results.tar.gz. It contains six
new runs plus completed original2e-6 controls and previous1e-6/4e-6 references:
CSVs, launch/effective configs, runtime/optimizer/source/launch/completion metadata,
logs and actual split evidence. Weights are excluded by an explicit file allowlist.
The previous run_local_lr outputs must remain available to verify/package references.
No reference is launched by this script. Packing refuses an existing archive.

Default Python /home/sll/anaconda3/envs/WZX-AI/bin/python; ABLATION_PYTHON overrides.
HF_HUB_OFFLINE1 defaults to cached CLIP. Full training refuses CPU. check-configs
validates all six without source/data, while all --dry-run verifies the server
source/control and prints all six planned configs without training/writing.
No GPU index is assigned. Scripts do not promise bitwise training reproduction.

Use validation evidence for tuning priorities, report all four correlations and
all outcomes. These splits/test scores have already been viewed, so this is
exploratory tuning; select a candidate for seed replication rather than claiming
stable superiority from tiny single-seed differences. Do not mix dataset-specific
winners into a uniform method. No new real-data training was performed in cloud.

CPU tests cover actual optimizer isolation/LR curves, unchanged tensor/group
membership/other LRs/decay, fixed configs, effective-LR provenance rejection,
no-overwrite, exact six-run dispatch, failure propagation, preflight refusal and
weight-free packaging with reference evidence.
