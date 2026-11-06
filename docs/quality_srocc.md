# Quality Soft-SROCC loss-weight control

Fix train seed42, 1K subset1078 split3407 and 3K/2023 split42. Reuse the accepted
epoch4 projection_only_v1 source, SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
No pretraining, resume, seed/split change or source substitution.

Change only train.lambda_srocc_quality relative to each dataset's baseline:

| Dataset | Control | qsrocc050 (x0.5) | qsrocc200 (x2) |
|---|---:|---:|---:|
| 1K subset1078 | 0.03 | 0.015 | 0.06 |
| 3K | 0.02 | 0.01 | 0.04 |
| 2023 | 0.02 | 0.01 | 0.04 |

Keep quality pairwise ranking coefficient0.03, MSE coefficient1, Soft-SROCC tau0.02,
and all Align loss coefficients unchanged. The original per-dataset coefficient
choices are retained as controls; candidates apply the same relative multiplier,
not individually selected weights. This adjusts how much minibatch soft ranking
contributes, not the evaluation SROCC computation or validation selection rule.

Restore all original optimizer settings: CLIP/ResNet2e-6, QA/PI/PT/quality1e-4,
semantic_proj in fusion1e-5, decay0.002. Alpha inherits, beta starts0.50. Scoring
and other dropout remain the original0.20/default explicit per-module values;
no score_dropout override is enabled. Preserve architecture, all other losses,
original train/val/test, 60 epochs and maximum validation Quality SROCC selection.
Do not stack any unsuccessful or pending earlier tuning candidates.

Configs are generated deterministically from accepted baseline YAMLs into fresh
launch_config.yaml files. Effective optimizer groups and actual memberships are
recorded, and source/control/completion/config/checksum provenance is checked.
The trainer already logs train_train_srocc_quality and its weighted contribution.
After completion and before packaging, all60 epoch rows must be present once;
weighted Soft-SROCC must equal the requested coefficient times raw Soft-SROCC
(relative tolerance1e-3, absolute1e-7 for AMP rounding and reductions), while MSE and quality ranking
contributions must still match their original coefficients. The loss implementation
and trainer are unchanged by this workflow.

Server setup and six explicit downstream runs:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_quality_srocc.sh all
```

all preflights every output path before launching, then runs both candidates on
1K,3K,2023 in order, stopping at the first failure. Existing outputs/CSVs/archives
are refused; partial failures remain and are not silently resumed. Do not rerun
all after partial completion; use the unstarted individual commands instead:

```bash
bash scripts/run_quality_srocc.sh qsrocc050 agiqa1k && bash scripts/run_quality_srocc.sh qsrocc200 agiqa1k
bash scripts/run_quality_srocc.sh qsrocc050 agiqa3k && bash scripts/run_quality_srocc.sh qsrocc200 agiqa3k
bash scripts/run_quality_srocc.sh qsrocc050 aigciqa2023 && bash scripts/run_quality_srocc.sh qsrocc200 aigciqa2023
bash scripts/run_quality_srocc.sh pack
```

After all six complete, upload
projection_only_v1_quality_srocc_1ksplit3407_results.tar.gz. It includes exact same-
split controls plus candidates, CSVs/config snapshots/logs, source/runtime/optimizer/
launch/completion metadata and actual membership evidence, without model weights.
No control or pretraining is rerun. No GPU index is selected.

Default Python /home/sll/anaconda3/envs/WZX-AI/bin/python; ABLATION_PYTHON overrides.
HF_HUB_OFFLINE1 uses the verified CLIP cache. CPU full launches are refused.
check-configs needs no source/data. all --dry-run verifies source/control and prints
six configs without writing or invoking trainers. CPU tests use a small differentiable
model with the real compute_losses and fake runner artifacts. No new real-data
training or server source-weight loading was performed in cloud.

Compare all four Quality/Align correlations and preserve all outcomes. Use validation
for tuning priorities, retain the original controls, and avoid combining dataset-
specific winners into a claimed uniform setting. These are exploratory evaluations
on previously viewed splits, not independent confirmation or multi-seed robustness.
