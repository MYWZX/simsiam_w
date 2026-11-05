# ResNet local-backbone LR control on the selected exploratory split protocol

Fix train seed42, 1K subset1078 split3407, 3K/2023 split42 and the accepted epoch4
projection_only_v1 component-anchor-selected source, SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
No pretraining, resume, source or split change is launched.

Change only train.layerwise_lr.local_encoder_mult: control0.02 (initial2e-6),
llr001=0.01 (initial1e-6), llr004=0.04 (initial4e-6), at base LR1e-4. This affects
backbone.local_encoder.* including its existing local_encoder.proj. It does not
change the independent P0 local_patch_proj, which remains in the quality group.
There is no local-stage, projection shape, initialization or BN-policy change.

CLIP remains2e-6, semantic_proj in fusion remains1e-5, PI/PT and QA remain1e-4.
The quality group including alpha/beta scalars and local_patch_proj remains1e-4.
Alpha is inherited, beta starts0.50 and learns, decay stays0.002. Keep architecture,
all losses, sample memberships, separate train/val/test, 60 epochs and maximum
validation Quality SROCC selection. Neither earlier unsuccessful hyperparameter
candidate nor another downstream checkpoint is used as a new source.

The unchanged cosine scheduler floor is1e-6: llr001 stays1e-6 throughout, while
llr004 decays4e-6 to1e-6. Other parameter-group schedules are unchanged. This is an
unproven local feature adaptation hypothesis, not a promised improvement.

Configs are generated deterministically from accepted baseline YAMLs and stored
as launch_config.yaml under fresh output dirs. Only the local multiplier, selected
1K split protocol, evidence flags and run identifiers differ. Reuse completed 1K
split3407 and 3K/2023 split42 controls. Source completion/config/mode/SHA256 and
matching controls are verified before launch. Actual optimizer settings before
scheduler steps are recorded in metadata/CSV and validated after completion.
Actual split indices and identity hashes are recorded; where historical control
membership evidence exists it must match the candidate exactly during packaging.

Server commands, two explicit candidates per dataset:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_local_lr.sh llr001 agiqa1k && bash scripts/run_local_lr.sh llr004 agiqa1k
bash scripts/run_local_lr.sh llr001 agiqa3k && bash scripts/run_local_lr.sh llr004 agiqa3k
bash scripts/run_local_lr.sh llr001 aigciqa2023 && bash scripts/run_local_lr.sh llr004 aigciqa2023
bash scripts/run_local_lr.sh pack
```

The wrapper defaults to server Python /home/sll/anaconda3/envs/WZX-AI/bin/python
(override ABLATION_PYTHON), HF_HUB_OFFLINE1 cached CLIP weights. Full cloud CPU runs
are refused. No implicit pretraining/all-dataset launch or GPU index assignment.
Existing outputs, CSVs and archives are refused; failed partial outputs remain.
check-configs validates six settings without source/data. --dry-run verifies source
and control and prints effective cfg without launching a trainer or writing files.

Upload projection_only_v1_local_lr_1ksplit3407_results.tar.gz after all six runs.
It includes same-split controls and candidate CSVs/configs/logs, source/runtime/
optimizer/launch/completion metadata, and actual split evidence. Explicit file
allowlists exclude all weights. Compare Quality/Align SROCC and PLCC, preserve all
outcomes, and do not assemble per-dataset winners into a unified baseline. These
are exploratory evaluations on already viewed splits, not multi-seed robustness.

CPU tests verify actual optimizer isolation, unchanged membership/model tensors/
other group LRs/decay, schedule endpoint, effective-LR provenance rejection, fixed
config/split/source policy, CPU refusal, no-overwrite and weight-free packaging.
No new server checkpoint loading or real-data training was performed in cloud.
