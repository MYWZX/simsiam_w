# Global PI: validation selection comparison

The existing quality-SROCC refits are controls. New alternatives use quality PLCC
or joint quality/alignment SROCC, with no change to training loss, pretrained
source, data membership, learning-rate schedule or model. The joint score is the
existing implementation: correlations mapped to [0,1], harmonic mean, mapped back.
Without alignment labels (1K), it reduces to quality SROCC. Lowest validation
regression loss is reported as a read-only reference; no loss-selected refit is
launched by this workflow.

Validation-only analysis of the uploaded completed development CSVs:

| Contexts | Dataset | Quality SROCC epoch | Quality PLCC epoch | Joint epoch | Lowest regression-loss epoch |
|---|---|---:|---:|---:|---:|
| 4 | subset1078 | 36 | 36 | 36 | 58 |
| 4 | 3K | 47 | 48 | 50 | 53 |
| 4 | 2023 | 29 | 44 | 40 | 40 |
| 5 | subset1078 | 8 | 21 | 8 | 29 |
| 5 | 3K | 58 | 43 | 44 | 43 |
| 5 | 2023 | 52 | 58 | 60 | 60 |

Development CSVs only have test metrics for the originally selected model.
Do not treat its test score as another epoch's score. New unmerged test scores
cannot be reconstructed without the corresponding weights. The workflow instead
runs fresh merged refits with validation-derived budgets.

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_globalpi_selection_comparison.sh --output outputs/globalpi_selection_comparison.json
```

This does not repeat pretraining or development training. It analyzes both context
counts and three datasets, reports quality SROCC/PLCC/joint/loss budgets, reuses the
existing quality-SROCC refit, and runs the distinct PLCC/joint budgets. With the
uploaded trajectories there are nine new merged refits. Identical budgets within
the same model/source/config are explicitly reused, with paths printed/reported;
no copied or invented test scores. Sources differ between context counts.

Analysis only (no training):

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/compare_globalpi_selection.py
```

Limit new runs or continue a remaining policy:

```bash
bash scripts/run_globalpi_selection_comparison.sh --tokens 5 --dataset agiqa3k --policy joint
```

Separate plans: `outputs/globalpi_selection_plans/{globalpi,globalpi_ctx5}`.
New run names append `_quality_plcc` or `_joint` to the original refit80 name.
Source fingerprints/config/CSV checks remain active. Existing result directories
are never overwritten. Valid existing plans without run outputs can be resumed;
partial/completed run outputs stop execution, so choose a remaining dataset/policy.

A fixed batch budget is an experimental stopping rule, not proof of convergence.
The final epoch can be partial and still complete that rule. Training for more
steps or full epochs is a separate budget experiment. Lowest regression MSE need
not maximize correlation: MSE measures absolute score error, SROCC measures order,
and PLCC measures linear association. Current val_reg_loss combines quality and
alignment MSE when alignment labels exist; it is not the entire training objective
including every auxiliary term. See evaluate() in train_cm_iqa.py.

No new real-data/GPU training has been performed in the cloud.
