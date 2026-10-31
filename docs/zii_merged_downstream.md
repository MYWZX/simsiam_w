# Fused-z I-I: downstream train+validation refit

This experiment reuses the completed z-I-I development runs and their shared
`outputs/richhf18k_zii_noce_noneg_pretrain_dev_seed42/best_component_anchor_pretrain.pth`.
Pretraining is unchanged and is not merged or repeated. Quality text CE and
legacy negative losses remain disabled; other scientific settings are unchanged.

The uploaded development results select the following budgets using validation
quality SROCC alone. Test metrics do not determine the budget.

| Dataset | Development epoch | Processed training batches |
|---|---:|---:|
| AGIQA-1K subset1078 | 23 | 2162 |
| AGIQA-3K | 30 | 7800 |
| AIGCIQA2023 | 44 | 9240 |

Plans are reconstructed and verified on the server; these values are not hardcoded.
The refit begins afresh from the same pretrained representation, merges the original
train and validation memberships, and retains original test membership. 1K uses
863 training images and 215 test images; its normalization is refitted on train+val.
Original learning-rate schedule units are retained. Each run ends at its fixed batch
budget and evaluates test once, without validation or best-epoch selection.

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_zii_merged_downstream.sh
```

No GPU index is assigned. `ABLATION_PYTHON` overrides the usual server Python.
To run one dataset: `bash scripts/run_zii_merged_downstream.sh agiqa3k`.
To run only its refit after a plan was already successfully created:
`bash scripts/run_zii_merged_downstream.sh agiqa3k refit`.
The script stops on existing plans or outputs instead of overwriting them. If an
earlier dataset finished, select the remaining datasets individually.

Plans: `outputs/zii_refit_plans/{agiqa1k,agiqa3k,aigciqa2023}.{json,yaml}`.
Result CSVs:

- `outputs/agiqa1k_subset1078_zii_noce_noneg_refit80_seed42.csv`
- `outputs/agiqa3k_zii_noce_noneg_refit80_seed42.csv`
- `outputs/aigciqa2023_zii_noce_noneg_refit80_seed42.csv`

No new real-data training was performed in the cloud environment.
