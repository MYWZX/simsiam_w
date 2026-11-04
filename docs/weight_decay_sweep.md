# Downstream weight-decay control from the accepted baseline

After the QA alpha-start comparison showed no unified improvement, retain the
accepted baseline unchanged. This next independent sweep changes only downstream
AdamW weight_decay: baseline 0.002 versus candidates wd005=0.005 and wd010=0.01.
Both candidates use the same values on all three datasets; no dataset-specific
winner is silently assembled into a unified method.

Source: the accepted projection_only_v1 epoch-4 component-anchor-selected
pretraining checkpoint, SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
The script verifies original completion, source mode/config/selector and pinned
fingerprints in docs/current_baseline.json. No pretraining, old-mode source,
downstream checkpoint initialization or resume is launched. The completed baseline
is reused as the control. QA alpha is inherited normally, not reset to 0.05/0.20.

Configuration checks allow only train.weight_decay and new output identifiers to
differ from configs/projection_only_v1_baseline. Seed/split seed 42, subset1078,
5 context tokens, g-only routing, PI/PT and local projection semantics, quality
losses/anchors/ordinal, original splits, 60 epochs and maximum validation Quality
SROCC selection are unchanged. No train+val merge. PI/PT initial LR stays 1e-4;
semantic_proj stays in the fusion group at 1e-5. Existing AdamW applies the configured
decay to all its parameter groups, including bias/normalization/scalar parameters;
this sweep preserves that grouping policy and changes no exclusion rules.

Motivation is the observed training MSE decrease with validation MSE plateau, not
proof that greater decay improves generalization. Decide tuning priorities using
validation results; retain all test outcomes and report the exploration honestly.
Do not keep narrowing the grid to chase the already observed test scores.

Server commands:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_weight_decay.sh wd005 agiqa1k && bash scripts/run_weight_decay.sh wd010 agiqa1k
bash scripts/run_weight_decay.sh wd005 agiqa3k && bash scripts/run_weight_decay.sh wd010 agiqa3k
bash scripts/run_weight_decay.sh wd005 aigciqa2023 && bash scripts/run_weight_decay.sh wd010 aigciqa2023
bash scripts/run_weight_decay.sh pack
```

Each training command names one candidate and one dataset; no implicit batch of
training or pretraining occurs. Defaults: server Python
/home/sll/anaconda3/envs/WZX-AI/bin/python (override ABLATION_PYTHON), offline cached
CLIP loading via HF_HUB_OFFLINE=1. check-configs validates configs without source
weights; candidate --dry-run verifies the server source but launches no trainer.
Full training refuses CPU. Existing output directories, CSVs and archives are
preserved by refusal; failed runs are not resumed implicitly.

Packaging requires all six completed candidate runs. The exclusive-create archive
projection_only_v1_weight_decay_results.tar.gz contains candidate CSVs, snapshots,
logs, launch/runtime/completion metadata, source fingerprints, configuration files,
and the original accepted baseline evidence for comparison. Files are explicitly
allowlisted; no model weights are included. Compare Quality and Align SROCC/PLCC
against the current baseline; 1K has no Align labels.

Cloud tests use small substitute encoders and fake runner artifacts; they check
config isolation, unchanged optimizer group membership/LRs/parameters, actual decay
values, CPU refusal, provenance rejection, no-overwrite and weight-free packaging.
No new real-data training or server checkpoint run was performed in cloud.
