# Learnable global/local quality score-mixing beta starts

Keep train seed42, accepted projection_only_v1 pretraining source, alpha inheritance,
decay0.002, all group LRs, architecture/losses, original 60-epoch budget and maximum
validation Quality SROCC selection. Change only the learnable score-mix beta start
from the control's 0.50 to beta025=0.25 or beta075=0.75.
Score = beta * local_quality + (1-beta) * global_quality. quality_mix_init stores
logit(beta), so values are -1.0986122886681098 and +1.0986122886681098.
This beta is not QA's feature residual alpha or quality_feature_mix_logit.

The user fixed 3K/2023 split42 and wrote 1K split3047; completed 1K results instead
exist for split3407. The number has not been confirmed. Therefore the runner has
NO implicit 1K split default and requires explicit --split-seed 3407 or 3047.
3K/2023 reject split overrides. No new unified baseline designation is made here.

With 1K split3407, reuse the completed splitcheck control (beta start0.50). With
new split3047, the matching control does not exist and must be explicitly run
first; the candidate launcher never starts it automatically. A split3047 control
uses the same audited subset1078 and independent explicit membership manifest,
755/108/215 counts and train-only scaling. It never replaces split3407 or split42.
3K and2023 reuse their completed accepted split42 controls, without rerunning.

Configurations are deterministically derived from accepted baseline YAMLs and
saved under each fresh output directory as launch_config.yaml before the trainer
starts. Only beta start, selected 1K membership revision/seed, evidence flags and
unique output IDs differ. Full retained representations are loaded as usual;
representation scope deliberately excludes the final score-mixing logit, so beta
survives loading. Post-load logging checks its actual value and trainability,
records a quality_mix_initialization event/metadata, and never mutates weights.
Legacy configs without the evidence flag preserve behavior.

Every run verifies the pinned source SHA256
fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686
and matching control completion. Output files are exclusive-create; failures are
preserved, not resumed or overwritten. Cloud CPU full launches are refused.
Cached CLIP offline mode defaults to1, Python defaults to the server WZX-AI path;
ABLATION_PYTHON can override. No pretraining or GPU number selection occurs.

Server commands for the already completed split3407 option, ONLY if confirmed:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_beta_start.sh beta025 agiqa1k --split-seed 3407 && bash scripts/run_beta_start.sh beta075 agiqa1k --split-seed 3407
bash scripts/run_beta_start.sh beta025 agiqa3k && bash scripts/run_beta_start.sh beta075 agiqa3k
bash scripts/run_beta_start.sh beta025 aigciqa2023 && bash scripts/run_beta_start.sh beta075 aigciqa2023
bash scripts/run_beta_start.sh pack --split-seed 3407
```

If split3047 is intentional, first explicitly run:

```bash
bash scripts/run_beta_start.sh control agiqa1k --split-seed 3047
```

Then use the same 1K/pack commands with3407 replaced by3047. Do not run both split
options and pick the winner. Default parser requires a dataset for each run and
an explicit 1K split for packaging. check-configs constructs all allowed settings
without data or source checks. Candidate --dry-run verifies source/control and
prints the full effective config without writing files or launching a trainer.

Packaging requires all six selected candidates plus their exact-split controls.
Files: projection_only_v1_beta_start_1ksplit3407_results.tar.gz or corresponding
1ksplit3047 name. Includes full config snapshots, CSVs, logs, runtime/launch/
completion metadata, source fingerprints and actual data split manifests; weights
are allowlisted out. When control membership evidence exists, candidate/control
sample order and indices must match exactly. Historical split42 controls did not
store actual membership manifests; their config/completion evidence is retained,
while new candidates record actual seed42 permutations.

Compare all four correlations to their corresponding same-split control. Since
1K's split is selected after viewing earlier test results, this is an exploratory
protocol; preserve prior split sensitivity results and do not call a split change
a parameter gain. Do not mix candidates per dataset into a unified result or claim
multi-seed robustness from this train-seed42 study. No real training ran in cloud.
