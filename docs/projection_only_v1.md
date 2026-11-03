# Projection-only v1 (independent code/experiment version)

Opt in with `model.fusion.mode: projection_only_v1`. Omitted mode defaults to
`attention`: historical configuration behavior and state keys remain unchanged.
Independent configurations are in `configs/global_pi_ctx5_projection_only_v1`.
They retain the matched baseline downstream checkpoint source and use distinct
output paths. No training has been run with this version; old results belong to
the old attention-computing code, not this version.

The new mode retains `backbone.fusion.semantic_proj` as the original Linear,
including bias, dimensions and operations. This name deliberately preserves
checkpoint keys and the fusion optimizer group in pretraining and downstream.
It is the global projection, not cross-attention. Current downstream initial LR
remains 1e-5; PI/PT's multiplier does not affect it. P0 still uses the independent
`local_patch_proj`, and the local encoder's own projection is unchanged.

The new module contains only semantic_proj. Image encoding returns `z=None`,
never a substituted g. Joint pretraining requires dual-quality g-only routes and
three-view/separate-clean supervision; pure SimSiam, z routes, fusion gates and
Final/fusion auxiliary losses are rejected. Legacy attention mode remains
available for those experiments. Projection-only is intended for the current
joint pretraining and downstream framework, not arbitrary legacy paths.

Downstream loading uses the existing `load_pretrained_backbone` with
`load_scope=representation`. Every selected target representation tensor is
required with matching shape; missing projection, QA, PI/PT, local-quality or
other selected representation keys abort before loading. Known obsolete
local_proj/attn/norm/ffn fusion tensors are explicitly listed when ignored.
The original semantic_proj keys load unchanged, with no conversion of weights.
Scoring heads remain freshly initialized under the existing representation
transfer policy, not silently inherited from another experiment. Do not use an
old full optimizer/resume checkpoint as an interchangeable new-mode resume.

Audit reports global_projection, cross_attention and local_projection separately,
with present/active flags (active means trainable parameters exist, not that the
objective uses them). local_projection means P0's local_patch_proj, not Fusion's
obsolete local_proj or the ResNet projection. The legacy aggregate is retained.
Cross-attention reports present=0 and active=0 in the new mode.

Removing attention and dropout changes initialization and training RNG consumption.
Eval parity requires identical retained weights, heads and input. There is no
promise of bitwise training reproduction or of unchanged real-data performance.
CPU tests use small substituted encoders with real heads/projections; the uploaded
result archives contain no baseline checkpoint tensors. Actual 1.27GB server
checkpoint loading and real-data evaluation remain unverified in cloud.

Validation: `OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
/workspace/.venvs/simsiam/bin/python -m unittest discover -s tests` passed all
245 tests in the CPU cloud environment. New tests cover retained tensor equality,
eval three-view output parity, missing-key rejection, absence of attention,
separate audit flags, both optimizer paths and all four new configurations.
