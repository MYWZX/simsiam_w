# Accepted baseline: split-seed sensitivity with train seed42

Fix downstream training seed=42, vary split_seed=123 and 3407. Reuse the same
accepted seed42 pretraining checkpoint (epoch4, Component Anchor validation SROCC):
SHA256 fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686.
No pretraining, resume, hyperparameter changes or baseline rerun. The prior
training-seed replication (123/3407 with split42) is a different experiment.

Configurations under configs/baseline_splitcheck preserve all accepted baseline
settings except split_seed, unique output IDs, and an opt-in split evidence flag.
The 1K configs explicitly select a new manifest revision for the same audited
subset1078 CSV. No excluded conflicting rows are restored. Original seed42
preserve-members manifest and all historical configs retain their old behavior.
New 1K memberships come from torch.randperm(1078) with the requested split seed;
counts stay train755/val108/test215, matching the baseline. The original seed42
manifest was derived from the original1080 split followed by two-row exclusion;
it is NOT replaced by a randperm1078 seed42 manifest.

The 1K preparer first verifies the existing audited subset, then writes a separate
revisions/subset1078_resplit_v1/split_seed{seed}.json. Existing different manifests
are refused rather than overwritten. Missing explicit manifests cannot silently
fall back to random splits. The loader verifies unchanged CSV SHA and exact
expected membership. Train-only min/max normalization is refit on each new train
split; val/test labels do not set it. Original train/val/test remain separate.
3K and 2023 use their original ratio-based random split logic with the new seed.

The opt-in data_split_manifest.json records actual train/val/test indices, sample
order/test identity hashes, counts, train/split seed, and overlap with the original
seed42 test membership. The trainer compares actual loader indices against the
requested permutation and checks disjointness and full dataset coverage before
model construction. These files, plus metadata and CSV events, are packaged.
Original configs without the opt-in flag create no new split evidence files.

Keep 60 epochs and validation Quality SROCC maximum selection, decay0.002,
inherited trainable QA alpha, PI/PT initial LR1e-4, global projection LR1e-5 and
all existing architecture/losses. Do not use the unsuccessful tuning candidates.
No implicit all-dataset launch; each command names one split seed and dataset.

Server commands:

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
bash scripts/run_baseline_splitcheck.sh split123 agiqa1k && bash scripts/run_baseline_splitcheck.sh split3407 agiqa1k
bash scripts/run_baseline_splitcheck.sh split123 agiqa3k && bash scripts/run_baseline_splitcheck.sh split3407 agiqa3k
bash scripts/run_baseline_splitcheck.sh split123 aigciqa2023 && bash scripts/run_baseline_splitcheck.sh split3407 aigciqa2023
bash scripts/run_baseline_splitcheck.sh pack
```

Default Python /home/sll/anaconda3/envs/WZX-AI/bin/python; ABLATION_PYTHON overrides.
HF_HUB_OFFLINE=1 defaults to existing CLIP cache. CPU full runs refused; configs
can be checked without data via check-configs. Source fingerprints, completed
training, runtime train seed42 and requested split seed, snapshots and actual split
indices are verified before packaging. Output/CSV/archive files are not overwritten.
Failed partial outputs are preserved. New outputs use train42_split123/3407 names.
Archive: projection_only_v1_splitcheck_results.tar.gz, including six runs, original
baseline reference, logs/CSVs/configs/metadata/source and actual split manifests,
without model weights. No cloud real-data training has been run.

Different split seeds change train/validation/test image membership and the 1K
train-derived scaling, so scores are NOT paired improvements on the original test
set. Report this separately as fixed-training-seed split sensitivity. A summary over
splits42/123/3407 is descriptive and is not the training-seed variance from the prior
study; test-set overlaps also mean these runs are not independent datasets. Do not
select the highest-scoring split as a new baseline or retune per split.
