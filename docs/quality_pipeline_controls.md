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

## Same-trajectory selection comparison

`--compare-selection-policies` trains one model for exactly 60 epochs, disables
early stopping, and uses the single normal validation pass per epoch for both:

- `quality`: maximize `val_quality_SROCC`.
- `joint`: maximize `val_component_hmean_SROCC` using the existing mapped
  harmonic mean of Quality and Align correlations.

Strict improvement updates a policy; ties retain the earlier epoch. Both
values must be finite and at least two validation Align labels must exist.
This comparison is for 3K/2023, not Quality-only 1K. Do not combine it with
`--selection-metric`. The effective snapshot records both policies and the
60-epoch override before optimizer/scheduler construction. These source
configs already specify 60 epochs, so their cosine horizon is unchanged;
other configs are explicitly overridden to 60 (progressive schedules retain
the existing encoder-start offset). Losses, augmentation, BN behavior and
model structure are unchanged by this flag.

Best states are independent CPU tensor copies including state-dict metadata,
never downstream `.pth` files. If both policies select the same epoch they
share a snapshot and one final validation/test evaluation. Different epochs
require two CPU model snapshots; plan sufficient host RAM for Large models.
Test data is evaluated only after all 60 epochs and cannot affect selection.
No in-progress early stopping or per-epoch test selection is performed.

CSV keeps the existing header and ordinary-mode schema. Comparison mode adds
`selection_policies` to epoch metrics, emits `quality_selected` and
`joint_selected` with `best_epoch`, `best_selection_value`, `best_val` and
`best_test`, then one `final` record with `comparison_complete`, `last_epoch`
and both results under `selection_policies`. No single-policy `best_test`
event is emitted in comparison mode. If epochs coincide, both named records
contain the same measured result. Consumers must not count these as two runs.

`run_metadata.json` and the `run_metadata` CSV event record the input checkpoint
SHA256, Git commit/dirty status, Python/library/CUDA/cuDNN versions, GPUs and
key deterministic/TF32 settings. The input file is hashed before construction
and again after loading; changes abort the run. The metadata is not a full
archive of code, data, bootstrap CLIP caches or runtime state, and does not
prove bitwise reproducibility. Settings are recorded without enabling new
deterministic algorithms. A missing `final` event means the comparison did
not finish. An interrupted run loses its in-memory selected weights.

### Commands on the training server

First sync the `work` branch using `git fetch origin`, `git checkout work`,
then `git pull --ff-only origin work` from the project directory. Use fresh
output directories and CSV paths; do not overwrite earlier experiments.
The Large Component Anchor checkpoint should be the completed epoch46 file;
its actual digest is recorded at runtime. The existing pretrained checkpoint
includes pretraining PCGrad; this comparison does not change its history.

AGIQA-3K:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml --clip-large --seed 42 --text-context-tokens 4 --quality-text-blend 0 --quality-text-lambda 0.005 --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients --compare-selection-policies --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth --output-dir outputs/agiqa3k_quality_cliplarge_safeinput_noneg_dualselect_seed42 --results-csv outputs/all_results_agiqa3k_quality_cliplarge_safeinput_noneg_dualselect_seed42.csv
```

AIGCIQA2023:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml --clip-large --seed 42 --text-context-tokens 4 --quality-text-blend 0 --quality-text-lambda 0.005 --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients --compare-selection-policies --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth --output-dir outputs/aigciqa2023_quality_cliplarge_safeinput_noneg_dualselect_seed42 --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_safeinput_noneg_dualselect_seed42.csv
```

These commands were checked for CLI/config compatibility in the CPU development
environment; full dataset/GPU training was not run there.

## Current architecture and code review

See the [current framework and exact data-flow notes](figures/aigc_iqa_current_framework.md)
and the [2026-10-21 code review](audits/current_code_review_20261021.md).
The generated figure describes the active Large/residual-feature configuration;
optional modules are not all enabled together. The review documents remaining
compatibility, loading, memory and training-protocol concerns without changing
existing model behavior or historical results.
