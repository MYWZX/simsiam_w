# Isolated scoring-head dropout control

Use train seed42, 1K subset1078 split3407 and 3K/2023 split42. Reuse the accepted
epoch4 projection_only_v1 source SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
Keep CLIP/ResNet LR2e-6, QA/PI/PT/quality LR1e-4, semantic_proj fusion LR1e-5,
inherited learnable alpha, beta start0.50, decay0.002, all losses, dimensions,
60 epochs, separate train/val/test and maximum validation Quality SROCC selection.
None of the earlier LR/alpha/beta/decay candidates is stacked into this study.

The original model.heads.dropout0.20 also configures PI/PT, NSS/text gates and
Align gates. Changing it would not isolate score heads. The new optional
model.heads.score_dropout instead overrides only these four main scoring heads:
global_quality_head (2 dropout layers), patch_quality_head (1), align_global_head
(2), align_patch_head (2). Seven modules total. Candidates sdrop010=0.10 and
sdrop030=0.30; the completed same-split controls use0.20 and are not rerun.
The generic heads.dropout stays0.20, all other explicit dropout configs remain
unchanged, and no BN/freezing policy changes. Legacy configs without score_dropout
use the original generic value. Explicit score_dropout must be finite in [0,1).

Dropout has no checkpoint tensors; no layer indices, shapes, initialized weights
or RNG consumed during construction change. Old checkpoints remain structurally
compatible, with the complete existing representation transfer policy. Score heads
are still freshly initialized under that policy; the source remains the accepted
project checkpoint. Eval removes dropout, so same-input/weight eval outputs match;
training masks and resulting trajectories change intentionally. No claim of equal
training scores or improved generalization is made.

Configurations are deterministically generated as launch_config.yaml under unique
fresh directories. The trainer records actual scoring module names and p values
in metadata and a scoring_dropout_initialization CSV event; effective optimizer
groups and actual memberships are also recorded. Verification checks config/source/
selection/splits, group LRs/decay, actual scoring probabilities, completion hashes
and event consistency. Source/control verification runs before a launch. Where old
controls have membership evidence, it must match exactly; old split42 controls
retain config evidence while new candidates record the actual seed42 permutation.

Server setup and explicit all-six sequence:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_score_dropout.sh all
```

all checks every target path before launching, then runs both probabilities on
1K, both on3K, both on2023. Any failure stops and partial outputs remain. Do not
rerun all after partial completion; use only the unstarted individual commands:

```bash
bash scripts/run_score_dropout.sh sdrop010 agiqa1k && bash scripts/run_score_dropout.sh sdrop030 agiqa1k
bash scripts/run_score_dropout.sh sdrop010 agiqa3k && bash scripts/run_score_dropout.sh sdrop030 agiqa3k
bash scripts/run_score_dropout.sh sdrop010 aigciqa2023 && bash scripts/run_score_dropout.sh sdrop030 aigciqa2023
bash scripts/run_score_dropout.sh pack
```

Upload projection_only_v1_score_dropout_1ksplit3407_results.tar.gz once all six
complete. It contains same-split controls and candidates: CSVs/configs/logs,
runtime/optimizer/dropout/source/launch/completion metadata and split evidence,
without model weights (explicit file allowlist). Existing output/CSV/archive paths
are refused, not overwritten/resumed. No pretraining or GPU index selection.

Default Python /home/sll/anaconda3/envs/WZX-AI/bin/python; override ABLATION_PYTHON.
HF_HUB_OFFLINE1 uses the existing CLIP cache. Full CPU launches are refused.
check-configs needs no source/data; all --dry-run verifies source/control and prints
six configs without training/writing. Only CPU small-model/fake-artifact tests have
run in cloud; no new server source-weight loading or real-data training occurred.

Use validation evidence for tuning, retain all outcomes and compare all four
Quality/Align correlations. This remains single-seed exploratory tuning on already
viewed splits, not stable superiority or a per-dataset assembled new baseline.
