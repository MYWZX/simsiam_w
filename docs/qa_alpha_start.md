# QA residual alpha-start control from the accepted projection-only baseline

Two downstream-only candidates: alpha005 starts learnable alpha at 0.05;
alpha020 starts at 0.20. The completed current baseline (inherited pretrained
alpha) is the control and is never rerun. This studies the starting point of a
trainable residual coefficient, not a fixed coefficient or removal of QA.

Six independent configurations live under configs/qa_alpha_start. Each differs
from the corresponding configs/projection_only_v1_baseline file only in output
identifiers and train.post_load_qa_alpha. All three datasets share the same two
values. PI/PT initial LR remains 1e-4 and semantic_proj remains 1e-5. Original
splits, seed/split seed 42, subset1078, 60 epochs, quality-SROCC validation
selection, and all existing losses/architecture/other LRs are retained. No merge.

The source is the accepted epoch-4 best_component_anchor_pretrain.pth, SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
The runner verifies baseline completion, checkpoint cfg/mode/selection and the
pinned record in docs/current_baseline.json before every candidate. It never
launches pretraining or loads a downstream checkpoint. Full runs refuse CPU.
The wrapper defaults HF_HUB_OFFLINE=1 to reuse the verified server CLIP cache;
there is no fallback to random text or alternative source weights.

Only after complete representation loading, apply_post_load_qa_alpha overwrites
quality_adapter_gate_logit with logit(requested alpha), without consuming RNG or
replacing the Parameter object. QA weights, PI/PT, global/local projections,
quality-local representations and all other loaded tensors are preserved. Normal
representation transfer still freshly initializes the dataset scoring heads as
in the baseline. The override and loaded/applied scalar are recorded in a CSV
event and run_metadata.json; alpha remains trainable in the original group.
Changing model.quality_adapter.residual_init alone would be overwritten by the
loaded gate, so this explicit post-load control is necessary. Legacy configs
without train.post_load_qa_alpha keep their original behavior.

Server commands (each line handles two candidates on one dataset; any failure stops):

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_qa_alpha_start.sh alpha005 agiqa1k && bash scripts/run_qa_alpha_start.sh alpha020 agiqa1k
bash scripts/run_qa_alpha_start.sh alpha005 agiqa3k && bash scripts/run_qa_alpha_start.sh alpha020 agiqa3k
bash scripts/run_qa_alpha_start.sh alpha005 aigciqa2023 && bash scripts/run_qa_alpha_start.sh alpha020 aigciqa2023
bash scripts/run_qa_alpha_start.sh pack
```

All six candidates must complete for packaging. Output:
projection_only_v1_qa_alpha_start_results.tar.gz. It contains candidate CSVs,
config snapshots, launch/runtime/completion metadata, logs, source fingerprints,
and the accepted baseline evidence for comparison. No weights are included.
Existing output/CSV/archive paths are refused and partial failures preserved.
Default Python is /home/sll/anaconda3/envs/WZX-AI/bin/python; ABLATION_PYTHON
can override it. check-configs validates all six configs without source weights;
--dry-run on a candidate still requires the verified server baseline source.

Compare all four correlation metrics against the current unified baseline;
1K has no Align labels. Use validation results to make tuning decisions and do
not combine different candidate values per dataset into a claimed unified
baseline. One seed is exploratory evidence, not a robust superiority claim.
No new real-data training or server-checkpoint run was performed in cloud.
