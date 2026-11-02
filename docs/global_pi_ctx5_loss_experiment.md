# Five contexts: loss-selected pretraining and downstream

Independent version of the five-context global-only PI experiment. Both stages
now select by minimum validation loss. All training loss coefficients, model/data
settings and seeds remain unchanged. This changes both stages' selection rules,
so it is not an isolated downstream-only selector comparison.

- Pretraining: `pretrain_selection_metric: val_loss`, mode `min`. This is the
  existing pretraining evaluation loss, including the active weighted objectives.
  Main selection also controls early stopping (patience 15, max 50 epochs).
- Downstream: `selection_metric: val_reg_loss`, mode `min`, max 60 development epochs.
  In the current two-output mode this is lambda_quality * quality MSE plus
  lambda_align * alignment MSE (1K has no alignment labels), rather than the full
  training objective with ranking and all auxiliary terms.
- The source is now the MAIN `best_pretrain.pth` selected by minimum val_loss,
  NOT the separately tracked `best_component_anchor_pretrain.pth`.
- Pretraining retains its original train/val/test split. Each downstream performs
  development training then train+val fixed-budget refit from the same new source,
  preserving test membership. The refit budget comes from minimum validation
  regression loss; refit has no validation selection and tests only at the end.
- Five real-prompt context tokens in both stages; PI receives g, P0 bypasses PI;
  CE/negative remain off; 1K is subset1078. No old outputs are overwritten.

First synchronize and run ONLY pretraining:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_global_pi_ctx5_loss_experiment.sh pretrain
```

After it completes, run each downstream separately:

```bash
bash scripts/run_global_pi_ctx5_loss_experiment.sh downstream agiqa1k
bash scripts/run_global_pi_ctx5_loss_experiment.sh downstream agiqa3k
bash scripts/run_global_pi_ctx5_loss_experiment.sh downstream aigciqa2023
```

Each downstream command includes development, budget preparation, and merged refit.
The runner checks a completed five-context minimum-loss source before downstream.
If a plan already exists and only refit is needed, use `refit DATASET` instead.
Existing plans/output directories cause a stop rather than overwriting a run.

Source:
`outputs/richhf18k_globalpi_ctx5_loss_noce_noneg_dev_seed42/best_pretrain.pth`.
Downstream names include `globalpi_ctx5_loss_noce_noneg_dev_seed42` and
`globalpi_ctx5_loss_noce_noneg_refit80_seed42`, with the subset1078 prefix for 1K.
Plans are in `outputs/globalpi_ctx5_loss_plans`.
`ABLATION_PYTHON` overrides the standard server Python; no GPU ID is specified.
No new real-data/GPU training was performed in the cloud environment.
