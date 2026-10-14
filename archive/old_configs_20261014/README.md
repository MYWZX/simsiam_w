# Archived experiment configurations

These YAML files belong to historical ablations and are not part of the current
mainline workflow. They are retained for reproducibility only.

The current mainline configurations remain in `configs/`:

- `richhf18k_cm_pretrain_decoupled_quality_align.yaml`
- `agiqa1k_cm_iqa_decoupled_quality_align.yaml`
- `agiqa3k_cm_iqa_decoupled_quality_align.yaml`
- `aigciqa2023_cm_iqa_decoupled_quality_align.yaml`

The archived variants cover mirror augmentation, VMamba/fusion gates, sampling
and loss reweighting, pseudo-alignment, PCGrad, Final supervision, older
quality-anchor variants, and other historical protocol changes. They must not be
used for current comparisons unless a separate reproduction explicitly names
the archived configuration.
