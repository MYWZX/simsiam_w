# Quality pipeline audit — 2026-10-20

Audited baseline: `3ee7a11`. Inputs: current source and uploaded seed42 residual_feature/full-CLIP-L result archives. No datasets, server runtime, or large checkpoint tensors are available locally. This audit changes no training behavior.

## Findings ordered by next-action priority

### 1. Disabling perturbation does not disable label-changing training augmentation

`train_cm_iqa.py --disable-perturbation` sets only dataset.use_perturb=false and empties perturbation_strengths. Both image transform implementations still randomly crop, horizontally flip, and jitter brightness/contrast/color. The residual_feature result configs use the default `legacy` augmentation. AGIQA3K __getitem__ assigns `image=image1`, and compute_losses uses that augmented image with the original whole-image MOS. AIGCIQA2023 and AGIQA1K use BaseIQADataset._make_visual_inputs with the same legacy behavior.

This is an objective/design concern, not proof every augmentation hurts performance. Cropping can remove a scored artifact or a prompt-mentioned object; brightness/contrast changes can change perceived quality. Flipping can invalidate directional prompts. Run a uniform, optional whole-image resize-only downstream control, retaining original normalization and all other training settings. Do not silently change existing baselines. The quality_safe path exists for 1K/2023 but is not wired into the 3K dataset.

### 2. Legacy easy-negative training is still enabled, with an identical-prompt negative bug

AGIQA3K/AIGCIQA2023 configs retain train.lambda_neg=0.05. This activates a separate branch regardless of use_align_rank=false and lambda_align_rank=0. The loss cyclically shifts prompts, calls the whole model again, then enforces a positive margin. It does not mask equal or empty prompts.

Reproduction: `[same, same, different, another]` shifts to `[same, different, another, same]`; the first image receives its own true prompt as its negative. No single deterministic scoring function can satisfy the positive margin for identical image/text pairs. Training dropout may create artificial differences. Even valid negative forwards run the full visual and Quality paths, creating additional BatchNorm updates, including to the Adapter. Thus this legacy branch can affect Quality as well as Align.

Correctness fix: mask identical/empty prompt pairs before the negative forward, and skip it when none are valid. The intended no-legacy-negative experiment should additionally set lambda_neg=0. This is distinct from hard-negative and constructed-prompt ranking.

### 3. Auxiliary losses are numerically large compared with Quality MSE

At each residual_feature run's selected epoch, epoch-mean weighted training losses are:

| Dataset | Quality MSE | 0.005 × five-level CE | ratio CE/MSE | 0.05 × legacy negative |
|---|---:|---:|---:|---:|
| AGIQA1K | 0.001345 | 0.002962 | 2.20 | 0 |
| AGIQA3K | 0.000517 | 0.002455 | 4.75 | 0.002068 |
| AIGCIQA2023 | 0.000176 | 0.003214 | 18.3 | 0.001687 |

These are scalar-loss magnitudes, NOT gradient-norm measurements and not proof that one loss dominates encoder optimization. Align patch auxiliary contributes a further ~0.000949 on AIGCIQA2023. Measure shared-encoder gradient norms before claiming the five-level weight is small or changing several losses together. A smaller shared lambda is a possible subsequent controlled experiment, not an established correction. The user previously deferred the lambda=0 comparison.

### 4. Joint epoch selection does not maximize validation Quality

AGIQA3K/AIGCIQA2023 use val_component_hmean_SROCC. Uploaded residual_feature runs show:

| Dataset | selected epoch | selected val Quality SROCC | highest val Quality SROCC epoch | highest value |
|---|---:|---:|---:|---:|
| AGIQA1K | 14 | 0.884888 | 14 | 0.884888 |
| AGIQA3K | 31 | 0.905772 | 43 | 0.907115 |
| AIGCIQA2023 | 29 | 0.824352 | 8 | 0.833581 |

Joint selection is a legitimate two-task protocol, not a bug. It cannot be described as selecting the best Quality model. A uniform quality-selection control could select val_quality_SROCC on every dataset. Old per-epoch weights were not retained, so test scores for those alternate epochs cannot be reconstructed from logs. Do not select an epoch using test results.

### 5. AGIQA3K high-resolution crop flag was not implemented end to end

The CLI writes dataset.local_multicrop settings and prints a success message, but AGIQA3K's builder does not pass these settings, its constructor has no crop options, and samples contain no local_crops. Therefore compute_losses receives local_crops=None on 3K, even with --quality-highres-crops. A 3K attention run may use its new attention pooling over ordinary base-resolution tokens, but not the advertised 336px crops. Prior 3K high-resolution results must not be cited as evidence for/against actual high-resolution crops. 1K/2023 have separate working multicrop plumbing. Add runtime batch assertions or reject unsupported combinations before future crop experiments.

### 6. RichHF pretraining PCGrad is still enabled

Current pretraining YAML has train.pcgrad.enabled=true; the uploaded full Large pretraining run's config_json confirms enabled=true. The training loop invokes per-parameter _pcgrad_backward. Earlier descriptions/diagrams saying PCGrad was removed do not match this checkpoint's training. Existing checkpoint provenance must retain this fact. Turning it off is a new pretraining experiment, not a downstream correction.

### 7. CLIP QuickGELU compatibility needs explicit verification before fresh pretraining

The user previously supplied an OpenCLIP warning: model quick_gelu=false versus OpenAI pretrained tag quick_gelu=true. Neither visual nor text constructor explicitly requests force_quick_gelu. Verify actual modules under the server's installed OpenCLIP version. If confirmed, use a consistent activation setting for both towers and both stages in a NEW pretraining run. Do not silently change activation for existing checkpoints: tensor shapes would match while the function changes.

### 8. Evaluation loss logging contains confirmed errors

In evaluate(), the align-loss meter is created but never updated, so val/test align_loss is always zero. In two_output_only mode the Final mask is zero, and val/test reg_loss reports the masked synthetic Final loss, also always zero. PLCC/SROCC are computed independently and remain meaningful. Current explicit correlation selection is unaffected; a fallback to val_reg_loss would select on a constant zero, which is unsafe. Correct loss logging and fail explicitly for missing selection metrics before relying on loss-based model selection.

## Paths checked with no specific failure found

- Quality/Correspondence use mosz1/mosz3; /100 normalization is compatible with uploaded MAT ranges, with no clipping indicated.
- Train/validation/test use disjoint seeded index subsets. Fixed split_seed is retained across run seeds.
- Quality projector is remapped into the downstream Quality Adapter; context parameters and five-level image-to-text mapping are selected for representation loading.
- The bank still recomputes text features through frozen CLIP so class contexts receive gradients; score_blend=0 does not disable its auxiliary loss.
- Residual alpha, Adapter, class contexts and predictor parameters have optimizer routes. Targeted small-model forward/backward and loading tests exist; this audit does not establish completeness of a particular server checkpoint load.
- Global-only removes the explicit local score, not ResNet information in Fusion. Residual mode's adapter-only diagnostic is not a separately trained classifier.
- Freezing BN statistics applies only to the Quality Adapter and retains affine gradients; its uploaded experiment does not improve consistently.

## Recommended sequence, no new model modules

1. Preserve the existing uniform residual_feature baseline and historical scores.
2. Repair identical-negative masking and evaluation logging as correctness changes; document changed training behavior.
3. Implement one shared resize-only augmentation switch that is actually honored by all three dataset loaders; test it independently first, with every other setting held constant.
4. Separately disable the legacy lambda_neg objective on both labelled Align datasets (1K is already zero), evaluating Quality and Align together.
5. Only then audit actual per-objective shared-encoder gradients and consider one auxiliary-weight adjustment or a uniform validation-Quality selection control.
6. Defer new CLIP pretraining/PCGrad changes until downstream correctness and controls are settled.

No claim is made that these changes will close a particular SOTA gap; that comparison requires matching dataset/split/evaluation protocols and the reference results.
