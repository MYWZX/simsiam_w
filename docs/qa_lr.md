# QA projector learning-rate control on the selected exploratory split protocol

Fix train seed42, 1K subset1078 split3407 and 3K/2023 split42. Reuse the accepted
epoch4 projection_only_v1 component-anchor-selected source, SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
No pretraining, alternative source, resume, seed or split change is launched.

Change only train.layerwise_lr.quality_adapter_mult. The baseline fallback is1.0
at base LR1e-4, giving QA initial LR1e-4. qalr010 sets multiplier0.10 and QA LR1e-5;
qalr050 sets multiplier0.50 and QA LR5e-5. Both decay to the original1e-6 floor over
60 epochs. This controls quality_adapter.* parameter updates, preserving the
transferred QA representation more strongly. It is an unproven tuning hypothesis,
not evidence that QA needs slower adaptation.

Quality adapter residual alpha is not in that group: quality_adapter_gate_logit
belongs to the quality group and remains learnable at initial LR1e-4. The final
score-mix beta (quality_mix_logit) also stays in quality at1e-4, starting0.50.
Neither alpha nor beta is overridden. QA BN running-stat updates follow the
original policy. PI/PT initial LR1e-4, CLIP/ResNet2e-6, global projection fusion
LR1e-5, decay0.002, other group LRs and all losses/architecture remain unchanged.
Retain separate train/val/test, 60 epochs and maximum validation Quality SROCC.

Configs are deterministically copied from accepted baseline YAMLs with only the
QA multiplier, selected 1K split protocol, evidence flags and output IDs changed.
They are stored as launch_config.yaml in each fresh run directory. Reuse completed
1K split3407 and 3K/2023 split42 controls. Source/control completion, cfg/mode,
SHA256 and actual runtime seed, split and optimizer settings are checked. Actual
optimizer group names, LRs, decay and counts are recorded before scheduler steps.
Actual data_split_manifest.json permits candidate/control membership checks where
the historical control already has that evidence. Legacy configs remain unchanged.

Server commands, two candidates per dataset:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_qa_lr.sh qalr010 agiqa1k && bash scripts/run_qa_lr.sh qalr050 agiqa1k
bash scripts/run_qa_lr.sh qalr010 agiqa3k && bash scripts/run_qa_lr.sh qalr050 agiqa3k
bash scripts/run_qa_lr.sh qalr010 aigciqa2023 && bash scripts/run_qa_lr.sh qalr050 aigciqa2023
bash scripts/run_qa_lr.sh pack
```

No implicit candidate/dataset batch is launched. Default server Python is
/home/sll/anaconda3/envs/WZX-AI/bin/python; override ABLATION_PYTHON. Offline cached
CLIP mode defaults to1. Full runs refuse cloud CPU. Existing outputs/CSVs/archive
are preserved by refusal, including partial failures. check-configs validates all
six configs without data/source; --dry-run verifies source/control then prints cfg
without training or writing. No GPU index is assigned.

After all six candidates complete, upload
projection_only_v1_qa_lr_1ksplit3407_results.tar.gz. It contains same-split control
and candidate CSVs, configs, logs, runtime/launch/completion metadata, source hashes
and actual split evidence; explicit file allowlists exclude model weights. Do not
compare 1K to split42, or mix candidates per dataset into one claimed method.
Report all four correlations and keep failures. These are exploratory tests on
splits already viewed; choose tuning priorities from validation, not continual
narrowing around observed test scores. No multi-seed robustness is established.

CPU tests cover true optimizer group isolation and scalar ownership, unchanged
model tensors/group membership/decay/other LRs, schedule endpoint, effective-LR
provenance rejection, fixed configs/splits, no-overwrite and weight-free packaging.
No new real-data training or server source-weight loading was performed in cloud.
