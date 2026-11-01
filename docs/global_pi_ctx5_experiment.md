# Global-only PI with five learned prompt context tokens

This is an independent context-count comparison against
[the four-token global PI experiment](global_pi_pretraining.md).
Only `model.text_encoder.learnable_context_tokens` changes from 4 to 5, together
with distinct run names/output paths and the corresponding NEW pretraining source.
All three downstream datasets also use five context tokens, so the learned
five-token context can be transferred with its original shape. Do not reuse the
four-token pretrained checkpoint.

The disabled quality text bank's `class_context_tokens: 4` is retained unchanged:
it is a separate class-specific prompt setting, inactive here. It does not control
the five trainable context tokens attached to the real prompt text. Quality CE and
negative pushes remain disabled; Soft-SROCC and all other settings remain unchanged.

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_global_pi_ctx5_experiment.sh
```

Default execution runs new original-split pretraining, then for each downstream:
60-epoch development, validation-quality-selected budget planning, and a fresh
train+val fixed-budget refit. No test-based budget selection. 1K uses subset1078.
Pretraining is not train+val merged. Test membership remains unchanged.

New source:
`outputs/richhf18k_globalpi_ctx5_noce_noneg_dev_seed42/best_component_anchor_pretrain.pth`.

Downstream run prefixes:

- `agiqa1k_subset1078_globalpi_ctx5_noce_noneg`
- `agiqa3k_globalpi_ctx5_noce_noneg`
- `aigciqa2023_globalpi_ctx5_noce_noneg`

Each has `_dev_seed42` and `_refit80_seed42` outputs/CSVs. Plans are stored in
`outputs/globalpi_ctx5_plans`. Four-token outputs/plans are untouched.

Stages for continuing completed work:

```bash
bash scripts/run_global_pi_ctx5_experiment.sh pretrain
bash scripts/run_global_pi_ctx5_experiment.sh downstream
bash scripts/run_global_pi_ctx5_experiment.sh downstream agiqa3k
bash scripts/run_global_pi_ctx5_experiment.sh refit agiqa3k
```

`downstream` includes development, planning and refit. `refit` requires the existing
plan. Existing plans/results cause a stop rather than an overwrite; choose the
remaining dataset/stage to continue. `ABLATION_PYTHON` overrides server Python.
No GPU index is specified. No real-data/GPU training was performed in the cloud.
