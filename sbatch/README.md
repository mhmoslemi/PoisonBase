# PoisonBase SLURM jobs

This directory expands the grouped commands in `remaining_full_tex_commands.txt`
and `run_defense.txt` into one `sbatch` script per table cell:

- `attack/`: 129 jobs, one model/pair/attack/budget/selection cell per job,
  using 8 targets and 5 victims.
- `defense/`: 62 jobs, one model/attack/budget/selection/defense cell per job,
  using 7 targets and 5 victims.
- `remaining/`: one cache-building attack plus a dependency-aware submitter for
  the only two defense cells that did not complete.
- `missing_defense/`: 50 defense cells that are blank in the current
  `defense.tex` and were never part of the original 62-job set. This set
  deliberately excludes the two cells already handled by `remaining/`.
- `full_b20_b40/`: one-cell attack jobs for the Greedy, Greedy-J, and DPP-J
  blanks in the added ResNet20/VGG13 budget-20 and budget-40 rows of `full.tex`,
  using 8 targets and 6 victims per target.
- `manifest.tsv`: source command, estimated L40S runtime, and requested walltime.

Every job requests one L40S, one CPU core, and 7 GB of host memory. Walltime is
the configuration-specific L40S estimate plus 45 minutes. Standard cells
request at most 7 hours. Attack cells estimated to take more than 7 hours
are not capped: they request their full estimate plus 45 minutes and appear at
the end of `submit_attack.sh`. A timed-out cell can be submitted again and
resumes from its saved target/trial artifacts.

## Submission

From `/home/mmoslem3/scratch/attack_if`:

Submit the newly discovered, previously unsubmitted defense cells with:

```bash
sh submit_defense-missing.sh
```

This submits 50 independent one-cell jobs. If the earlier missing-cache retry
chain has not been submitted yet, submit it separately with:

```bash
sh submit_remaining.sh
```

The original `submit_attack.sh` and `submit_defense.sh` remain available for a
fresh full rerun; do not use them for the current remaining-work pass.

Submit an individual configuration with, for example:

```bash
sbatch sbatch/attack/attack_001_convnet_gradmatch_dog_bird_b0_001_ours_std.sh
```

`submit_remaining.sh` expresses this ordering with SLURM `afterok`
dependencies, so the two defenses cannot start until the missing random-poison
cache has been created and synced successfully.

## What is staged

The shared `_job_common.sh` copies only the required Python/configuration files,
the extracted CIFAR-10 directory, the configuration's pinned target file, and
configuration-specific caches/results:

- Attack: the selected model's surrogate and clean-victim cache directories,
  plus only the run directories named by that job.
- Defense: only the attack run directories containing the requested saved
  perturbations, the matching defended-victim cache, and only the defense run
  directories named by that job.

Runs execute under `$SLURM_TMPDIR/attack_if`. Per-run directories (including
logs, CSV, JSON, `.pt`, and poison-cache files) are synced back to
`/home/mmoslem3/scratch/attack_if`. Shared `summary_all.csv` files are not copied
back because concurrent jobs would overwrite each other's aggregate; the
per-run outputs remain complete and authoritative.

SLURM stdout/stderr goes directly to `sbatch/logs/`. A pre-timeout `USR1` signal
asks the wrapper to stop its job step and sync completed artifacts five minutes
before the allocation ends.

Regenerate after either command list changes:

```bash
python3 sbatch/generate_jobs.py
```

Regenerate only the newly blank defense-cell set after editing `defense.tex`:

```bash
python3 sbatch/generate_missing_defense.py
```

Regenerate and submit the added ResNet20/VGG13 budget-20/40 cells with:

```bash
python3 sbatch/generate_full_b20_b40.py
sh submit_full-b20-b40.sh
```

## Twenty-five-minute smoke tests

These two jobs use isolated `toy_result/`, `toy_defense_result/`, and
`toy_cache/` directories, so they cannot append to the paper results:

```bash
sbatch sbatch/toy/toy_attack.sh
sbatch sbatch/toy/toy_defense.sh
```

The attack uses one preselected target, one victim, one surrogate epoch, two
crafting steps, and one victim epoch. The defense replays one saved perturbation
from the existing ResNet20BN GradMatch+DPP dog--bird budget-0.002 run, using one
target, one victim epoch, and one FRIENDS noise-generation epoch.

The staging/sync pattern follows the Alliance guidance for node-local storage:
inputs must be copied into `$SLURM_TMPDIR`, and outputs must be copied back to
persistent storage before the job ends.
