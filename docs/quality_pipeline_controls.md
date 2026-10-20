# Downstream quality training controls

All experiments reuse the same CLIP Large Component Anchor pretraining checkpoint and the same residual_feature architecture on all three datasets. Keep normal BN updates, learnable residual alpha initialized to 0.1, patch/global score fusion, five-level CE 0.005, blend=0, seed42, and disable_perturbation. No new model modules are introduced.

- `--quality-safe-input`: whole-image resize and existing ImageNet normalization, without random crops, flips, or color jitter. Implemented for 1K/3K/2023. Does not replace or imply `--disable-perturbation`; use both for this control. Mirror augmentation is rejected if also enabled.
- `--disable-legacy-negative`: sets only lambda_neg=0. It does not affect the separate hard/constructed ranking options, which are already off in these configs.
- `--audit-loss-gradients`: logs first-training-batch unscaled gradient norms on Fusion parameters for weighted Quality MSE, weighted five-level CE, the whole Quality task, and the whole Align task. These are representative Fusion gradients, not whole-backbone gradients. It uses autograd probes without model forwards or modifying optimizer .grad. CUDA/AMP magnitudes should be interpreted as diagnostics, not exact full-dataset averages.
- `--selection-metric val_quality_SROCC`: optional independent selection-protocol control. Do not combine with the first input/loss comparison. Default joint selection is preserved for 3K/2023.

## Correctness changes applied to all new runs

Identical/empty shifted Prompt negatives are masked. When all pairs are invalid, the second full model forward is skipped. For partially valid batches the whole batch still runs to preserve BatchNorm batch size, and invalid pairs contribute no negative loss. Therefore new runs with lambda_neg>0 are not strictly identical to the historical unfiltered baseline.

Evaluation now accumulates real Align MSE. Two-output `reg_loss` is weighted Quality MSE plus weighted Align MSE, excluding rank and auxiliary losses; it is not the full training objective. Missing/nonfinite selection metrics raise errors instead of silently selecting on zero.

AGIQA3K now reads local_multicrop and actually returns local_crops. An enabled crop setting with missing/wrong-size batch tensors raises an error. The first new experiments below do NOT enable high-resolution crops.

Training CSV includes weighted_quality_mse/text/rank/srocc, weighted_align_patch and weighted_legacy_negative, alongside raw losses. Historical data and metrics are not rewritten.

## Minimal experimental sequence

A: Add quality-safe-input (plus optional audit-loss-gradients) to the existing uniform residual_feature command. Run all three datasets, using output suffix `residual_safeinput_seed42`.

B: Keep A and additionally disable-legacy-negative. Run 3K and 2023, using suffix `residual_safeinput_noneg_seed42`. For 1K lambda_neg is already 0; reuse its A result. This is the same B method on all datasets, with no changed structure or dataset-specific method choice.

A vs B isolates disabling valid legacy negatives on otherwise identical corrected code. A vs historical residual runs additionally includes the identical-negative correctness fix; do not attribute every difference solely to image augmentation. A strict augmentation-only estimate would require a fresh corrected-code legacy-input control.

Do not simultaneously change text lambda, epoch selection, Adapter BN, or architecture. Large pretraining need not be repeated.

## Pretraining provenance

Existing CLIP Large checkpoint was trained with PCGrad enabled. A new `--disable-pretrain-pcgrad` flag makes disabling it explicit for future NEW pretraining experiments. It does not modify existing checkpoints and cannot be applied to resume an otherwise identical run without changing its training protocol. Current initialization also prints PCGrad status.

The OpenCLIP QuickGELU compatibility concern remains a separate verification task for a new pretraining run. This change deliberately preserves existing activation functions/checkpoint semantics.
