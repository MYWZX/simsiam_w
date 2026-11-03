# Restore the best baseline's source and selection; PI/PT LR control

This supersedes loss-selected workflows for subsequent tuning. Historical loss
experiments remain preserved. The scientific control is the completed five-context
`globalpi_ctx5_noce_noneg` unmerged development baseline.

- Pretraining selection: Component Anchor validation SROCC, existing source selected
  at epoch 5, not the minimum-loss epoch-14 source.
- Downstream selection: maximum `val_quality_SROCC` over the existing 60-epoch budget.
- Original train/val/test split retained; no train+val merge or fixed-batch refit.
- Same architecture, five contexts, g-only PI/P0 bypass, losses and seeds.
- The ONLY training change is downstream PI/PT LR multiplier 1 -> .1 (initial
  lr 1e-4 -> 1e-5). Context learning rate and other parameter groups are unchanged.
- Source is `outputs/richhf18k_globalpi_ctx5_noce_noneg_dev_seed42/best_component_anchor_pretrain.pth`.
  The runner verifies the completed baseline configs and source fingerprint before
  launching each new downstream run. No pretraining is launched.

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_ctx5_srocc_projlr01.sh agiqa1k
bash scripts/run_ctx5_srocc_projlr01.sh agiqa3k
bash scripts/run_ctx5_srocc_projlr01.sh aigciqa2023
```

Results use `globalpi_ctx5_srocc_projlr01`, with independent paths and run IDs.
Existing results are never overwritten. No GPU ID is assigned. `ABLATION_PYTHON`
overrides server Python. 1K uses subset1078. No new real-data training was run in cloud.

Pack all three downstream results/configs/metadata and original source logs without
model weights:

```bash
bash scripts/run_ctx5_srocc_projlr01.sh pack
```

Upload `cliplarge_ctx5_srocc_projlr01_results.tar.gz`.
The unmodified best baseline (PI/PT multiplier 1) is already completed; use those
original results as controls rather than rerunning the same baseline.
