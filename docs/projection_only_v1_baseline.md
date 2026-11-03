# Independent projection_only_v1 baseline from fresh pretraining

This is a new baseline, not a relabeling of attention-mode results. Configuration
files live in `configs/projection_only_v1_baseline`. They match every setting in
the original best five-context baseline except the explicit projection-only mode,
independent output/run identifiers and the new downstream checkpoint source.
In particular PI/PT downstream multiplier is **1.0**, initial LR **1e-4**;
semantic_proj remains in the fusion group at **1e-5**. Seed/split seed are 42.
Both semantic image routes use g, P0 bypasses PI, and dual residual quality,
NSS, anchors/ordinal and the original other losses are retained. Quality text CE
and negative push remain disabled. AGIQA-1K uses subset1078 with preserved splits.

Pretraining uses original CLIP/OpenAI and ResNet initialization, official train/val,
no project checkpoint or resume, and maximum val_component_anchor_SROCC selection.
Downstream uses original train/val/test, 60 epochs, maximum val_quality_SROCC.
There is no merge/refit. No stage launches subsequent stages; no argument means
pretrain only. Existing output directories, CSVs and archives cause refusal.
Partial failures are preserved and cannot be resumed or silently reused by this
workflow. Full runs refuse CPU. Dry-run does not launch a trainer, but downstream
dry-run still requires a verified completed source. No GPU index is assigned.

The runner records runtime metadata (including code commit), launch config hash,
console log and completion hashes. Downstream verifies a single final CSV record,
no resume event, config snapshot, selected checkpoint epoch, representation scope,
checkpoint cfg/run_id/mode, expected parameter prefixes, absence of legacy attention
weights and SHA256 before launch. It stores source_fingerprint.json and checks the
trainer's post-load fingerprint after completion. Thus the source is the newly
completed projection-only run, not the old baseline or loss-selected run.
Hashes detect changes to completed artifacts; they are provenance checks, not
independent proof of training correctness. Dataset content is not fingerprinted.

Server commands (each runs only the named stage):

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_projection_only_v1_baseline.sh pretrain
bash scripts/run_projection_only_v1_baseline.sh pack-pretrain
bash scripts/run_projection_only_v1_baseline.sh agiqa1k
bash scripts/run_projection_only_v1_baseline.sh agiqa3k
bash scripts/run_projection_only_v1_baseline.sh aigciqa2023
bash scripts/run_projection_only_v1_baseline.sh pack-downstream
```

Run only the first command initially. Run each later command separately when its
prerequisites are complete. Default interpreter is
`/home/sll/anaconda3/envs/WZX-AI/bin/python`; override via ABLATION_PYTHON.
Original pretrained CLIP/ResNet assets and datasets must be available on the server;
the runner preserves inherited network/offline settings and never logs in or
substitutes project weights for missing upstream assets.

Archives are `projection_only_v1_baseline_ctx5_pretrain_results.tar.gz` and
`projection_only_v1_baseline_ctx5_downstream_results.tar.gz` in the repository root.
They explicitly allowlist CSVs, configuration snapshots, runtime/launch/completion
metadata, console logs and source fingerprints, excluding all model weights.
The second archive requires all three downstream runs and includes source evidence.
Save archives elsewhere if another packaging pass is needed; no archive overwrite.

Removing attention/dropout changes RNG consumption. No claim of bitwise training
reproduction, equal scores or transferable historical results is made. CPU tests
use fake artifacts/subprocesses to verify orchestration and rejection behavior;
no new baseline training has been executed in cloud.
