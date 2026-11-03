# Downstream-only PI/PT learning-rate control

Control source: the completed five-context minimum-loss pretraining and downstream
baseline (`global_pi_ctx5_loss_noce_noneg`). This experiment changes only downstream
`train.layerwise_lr.shared_projection_mult` from 1.0 to 0.1, plus run identifiers.
At base lr=1e-4, PI/PT start at 1e-5 rather than 1e-4. Other groups, model structure,
BN statistics policy, losses, selection policy, seeds and split memberships are
unchanged. Source is the same `best_loss_representation.pth`; no new pretraining.

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_ctx5_projlr01.sh agiqa1k
bash scripts/run_ctx5_projlr01.sh agiqa3k
bash scripts/run_ctx5_projlr01.sh aigciqa2023
```

Each command runs development (60-epoch upper budget, minimum val_reg_loss
selection), derives a NEW fixed-batch plan from that trajectory, and refits from
the same pretrained source with train+val merged. The original test membership is
preserved and refit evaluates it only at the end. 1K is subset1078. Stopping partway
through the last refit epoch remains expected when the fixed budget is reached.

Existing outputs are not overwritten. If development is already complete, use
`bash scripts/run_ctx5_projlr01.sh DATASET plan`, then
`bash scripts/run_ctx5_projlr01.sh DATASET refit`. Only new matched plans apply;
never reuse a baseline plan after changing learning rate.

Pack all three datasets' results, source logs, configurations and plans (no weights):

```bash
bash scripts/run_ctx5_projlr01.sh pack
```

Upload `cliplarge_ctx5_loss_projlr01_results.tar.gz`. New names contain
`globalpi_ctx5_loss_projlr01`; plans are under
`outputs/globalpi_ctx5_loss_projlr01_plans`.
No GPU ID is specified. `ABLATION_PYTHON` overrides the server Python.
No real-data/GPU experiment was run in the cloud.
