# Downstream global-shared PCGrad

No re-pretraining is needed. Use the existing decoupled Quality checkpoint and
retain the Quality Adapter. Data, loss coefficients, initialization, and
validation model selection are unchanged.

## Modes

- Normal backpropagation: omit PCGrad flags (with an original YAML that has no
  PCGrad enabled).
- New implementation: `--enable-pcgrad --pcgrad-mode global_shared`.
- Legacy per-tensor variant: `--enable-pcgrad --pcgrad-mode per_parameter`.
- Old YAML snapshots with `enabled: true` but no mode still use the legacy
  variant. New CLI runs default to `global_shared`; use the explicit mode for
  reproducibility. The effective mode is saved to `config_snapshot.yaml`.

The new implementation treats all parameters reached by BOTH tasks as a
single shared gradient vector. A negative whole-vector dot product triggers
projection; a negative dot product on an individual tensor does not. Private
parameter gradients are unchanged. Missing-label/single-task batches use
ordinary summed-loss backward. The new mode requires the two-output protocol
without synthetic Final supervision.

## Projection strength (soft PCGrad)

Use `--pcgrad-mode global_shared --pcgrad-projection-strength 0.5` for the
single-variable half-strength ablation. The strength flag also enables PCGrad;
it does not change any loss coefficient, model module, dataset, or pretraining.

For shared gradients with a negative whole-vector dot product:

`g = (1 - eta) * (gQ + gA) + eta * gPCGrad`

- `eta=1` (default): exactly the previous full global-shared projection.
- `eta=0.5`: half the projection correction, NOT half either task's loss.
- `eta=0`: one ordinary summed-loss backward pass. Geometry is not computed;
  diagnostic active counts/norms/conflict rate are zero, NOT measured agreement.
- Nonconflicting shared gradients and all private gradients are unchanged.
- Single-task batches (e.g. 1K) use ordinary backward at every strength.

Strength must be finite and in `[0, 1]`. The legacy `per_parameter` mode
rejects this setting instead of silently ignoring it. Missing strength defaults
to `1`, preserving existing global-shared commands and YAML snapshots.
Partial projection need not remove the entire gradient conflict, and cannot
guarantee improved validation/test scores.

The same option is available in YAML, and its effective value is recorded in
`config_snapshot.yaml` and the CSV run configuration:

```yaml
train:
  pcgrad:
    enabled: true
    mode: global_shared
    projection_strength: 0.5
```

For the next comparison, retain `best_quality_pretrain.pth` from the existing
decoupled pretraining run and the original downstream configs. Start with seed
42 on both 3K and 2023. Use distinct output directories containing `eta05` so
the full-projection and ordinary-backprop results are not overwritten.

## Quality Adapter low-LR ablation

For the next single-variable experiment, add
`--quality-adapter-lr-mult 0.10` to the existing half-strength command.
This sets ONLY `train.layerwise_lr.quality_adapter_mult` in the effective
configuration; the original three downstream YAMLs remain unchanged.

- Quality Adapter initial LR: `1e-4 -> 1e-5` with the current base LR.
- Quality/Align prediction heads: still `1e-4`.
- Fusion: still `1e-5`; visual encoders: still `2e-6`.
- Keep `global_shared`, `projection_strength=0.5`, seed/split 42 and
  `best_quality_pretrain.pth` unchanged for the 3K/2023 comparison.
- No extra pretraining, new model module, loss change, freeze/detach or BN
  policy change. The existing cosine scheduler and absolute `min_lr=1e-6`
  remain unchanged, so the tenfold ratio describes the INITIAL LR, not every
  scheduled step.
- The flag requires enabled layerwise LR and an enabled, unfrozen Quality
  Adapter; invalid/nonfinite/nonpositive values fail before model/data loading.
- Omitting the flag preserves the existing fallback to `quality_mult=1.0`.
- The multiplier is included in `config_snapshot.yaml` and CSV `config_json`.
- Use distinct output directories such as
  `agiqa3k_decoupled_best_quality_pcgrad_global_eta05_qa_lr01_seed42`.
  For 1K the same setting lowers Adapter LR while PCGrad remains the ordinary
  single-task fallback; that is a genuine LR ablation, not a PCGrad effect.

## Diagnostics

The terminal prints the active step count, conflict rate, cosine similarity,
Quality/Align shared gradient norms, update norm ratio and nonfinite count.
Epoch rows in the existing results CSV contain the following `metrics_json`
keys (no extra checkpoints or predictions are saved):

- `train_pcgrad_projection_strength` (recorded even for zero/single-task runs)
- `train_pcgrad_active_steps`, `train_pcgrad_steps`, `train_pcgrad_active_rate`
- `train_pcgrad_conflict_steps`, `train_pcgrad_conflict_rate`
- `train_pcgrad_cosine`
- `train_pcgrad_quality_grad_norm`, `train_pcgrad_align_grad_norm`
- `train_pcgrad_grad_norm_ratio` (Quality / Align)
- `train_pcgrad_update_norm_before`, `train_pcgrad_update_norm_after`
- `train_pcgrad_update_norm_ratio` (after / before)
- `train_pcgrad_shared_numel`, `train_pcgrad_nonfinite_steps`

Conflict rate and gradient means are conditional on finite, nonzero shared
gradients for both tasks. No active steps (e.g. 1K) is NOT evidence that two
tasks agree. Norms describe the unscaled shared gradient before clipping and
the AdamW step; they are not the actual parameter-update norm. Exact summed
gradient cancellation has an undefined norm ratio, recorded as zero.

## Tests

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## Result reporting

A method does not need to improve every seed to remain an exploratory
candidate. Record the best single run and all attempted seeds' aggregate
results separately. A seed selected after examining test scores is an
exploratory best-of-seeds result, not an unbiased final benchmark estimate.
Fix method/seed selection using validation data before a fresh evaluation.
Do not change the pre-existing validation checkpoint criterion to test-set
performance or silently drop unsuccessful runs.
