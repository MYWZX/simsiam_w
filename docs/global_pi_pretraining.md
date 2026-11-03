# Global-only PI pretraining and downstream experiment

Independent experiment; old configs and default z image-text routing are retained.

Pretraining:

- I-I: g1/g2 -> shared PI -> DI.
- Image-prompt/image-answer: g -> PI -> AG -> DI.
- Clean global alignment: g -> PI -> AG.
- Local alignment: P0 -> AP, bypassing PI, including positive local supervision.
- Prompt: frozen CLIP + four contexts -> U -> PT -> AT; Answer retains its existing path.
- PQ remains independent. Global/local quality, anchors, ordinal, component MOS and
  NSS remain active. Quality text CE and negative pushes remain disabled.

Downstream:

- g -> PI -> AG; P0 -> AP; text -> U -> PT -> AT.
- Quality remains the dual-quality residual baseline. PI/PT BN statistics are frozen
  downstream as before. No other training hyperparameters are intentionally changed.

Semantic losses and pure-global quality no longer use fused z or train its
cross-attention path. The semantic_proj Linear inside Fusion still generates g
and receives quality and semantic gradients. This experiment
retains the module for compatibility but does not introduce a new fused-z loss.
ResNet is still trained by local quality and local alignment/positive region paths.

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_global_pi_experiment.sh
```

The default runs a NEW original-split pretraining, then each dataset's 60-epoch
unmerged development, derives its validation-quality-selected step budget and
refits from the NEW pretrained source using train+val. Original test membership
is preserved; final refit evaluates it once. 1K uses the independent subset1078.
Outputs use `globalpi_noce_noneg`, with separate `dev` and `refit80` identifiers.
No previous result or checkpoint is overwritten.

Stages: `pretrain`, `downstream` (all dev+plan+refit), or `refit` (existing plans).
Optional second argument selects one downstream dataset. Example:
`bash scripts/run_global_pi_experiment.sh downstream agiqa3k`.
Existing outputs/plans stop execution; use the appropriate stage/dataset to continue.

Context count 4 is a hyperparameter, not an architectural requirement. The encoder
supports 0..16, including 5. Changing to 5 should be a separate pretrain+downstream
experiment with matching context counts; four-context checkpoints are not a clean
initialization for a five-context transfer comparison. Five context tokens do not
represent five quality classes. Increasing them also reduces available prompt
length by one token. Keep four here to isolate the image routing change.

This changes several coupled routes to obtain consistent global-only PI inputs;
it is not an isolated test of image-text input alone.
