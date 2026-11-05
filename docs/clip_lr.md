# CLIP visual backbone LR control on the selected exploratory split protocol

Fix train seed42, 1K subset1078 split3407, 3K/2023 split42. Reuse the accepted
epoch4 projection_only_v1 component-anchor-selected pretraining source, SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
No new pretraining/resume or alternative seed/split is launched.

Control global CLIP visual LR is2e-6 (global_encoder_mult0.02 at base LR1e-4).
Candidates glr001 and glr004 change only that multiplier to0.01/0.04: initial
CLIP visual LR1e-6/4e-6. This means backbone.global_encoder only; the global
projection semantic_proj is still in fusion at1e-5. ResNet stays2e-6, PI/PT1e-4,
QA1e-4, and all other existing group LRs and AdamW grouping/decay remain unchanged.
Alpha inherits pretraining, beta starts at0.50 and learns, weight_decay0.002.
Keep original losses, architecture, train/val/test, 60 epochs and maximum validation
Quality SROCC selection. No merge, no dropout or quality-rank tuning in this study.

The unchanged cosine schedule has eta_min1e-6, so glr001 stays at1e-6 throughout;
glr004 decays from4e-6 to1e-6. Other groups retain their original curves. This is
an initial LR multiplier control, not a global change to scheduler endpoints.

Configs are deterministically derived from configs/projection_only_v1_baseline
and written under fresh output dirs as launch_config.yaml. Only global encoder
multiplier, selected 1K split protocol, evidence flags and run IDs differ. Reuse
existing split3407 1K control and accepted split42 3K/2023 controls. Do not compare
1K to the old split42 control or relabel historical results.

Before launching, verify pinned source hashes/completion/cfg and matching control.
The trainer records actual optimizer groups immediately after construction and
before scheduler steps: group name, initial LR, decay and parameter count. This is
saved in optimizer_initialization CSV event and run_metadata.json. Legacy configs
without the evidence flag preserve behavior. Completion/package checks validate
all recorded group LRs and original decay, plus training seed, actual memberships,
source and selection rules. All runs save actual data_split_manifest.json.

Server commands (two candidates per dataset, failures stop the line):

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_clip_lr.sh glr001 agiqa1k && bash scripts/run_clip_lr.sh glr004 agiqa1k
bash scripts/run_clip_lr.sh glr001 agiqa3k && bash scripts/run_clip_lr.sh glr004 agiqa3k
bash scripts/run_clip_lr.sh glr001 aigciqa2023 && bash scripts/run_clip_lr.sh glr004 aigciqa2023
bash scripts/run_clip_lr.sh pack
```

Default Python /home/sll/anaconda3/envs/WZX-AI/bin/python, override ABLATION_PYTHON.
HF_HUB_OFFLINE1 preserves the verified CLIP cache workaround. CPU full training is
refused, existing outputs/CSVs/archives are not overwritten, and partial failures
are preserved. check-configs validates all six configs without source/data;
--dry-run verifies source/control and prints config without training or writing.
No GPU index is selected and no candidate starts another candidate automatically.

Archive projection_only_v1_clip_lr_1ksplit3407_results.tar.gz requires all six
completed runs and contains exact same-split controls, logs, CSVs, effective cfgs,
optimizer/runtime/source metadata and actual split indices, without model weights.
Where historical membership evidence exists, candidate/control sample order and
indices must match. Old 3K/2023 controls have cfg evidence; new candidates save actual
seed42 membership evidence. Compare all four correlations, keep all outcomes, and
use validation rather than repeatedly searching the observed test scores. The
split protocol was selected after prior results, so all tuning remains exploratory;
do not claim multi-seed robustness or mix per-dataset candidates into one baseline.

CPU tests use small substituted encoders and fake artifacts; no new server-weight
loading or real-data training was performed in cloud.
