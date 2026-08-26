# PoisonBase SLURM jobs

This directory expands the grouped commands in `remaining_full_tex_commands.txt`
and `run_defense.txt` into one `sbatch` script per table cell:

- `attack/`: 129 jobs, one model/pair/attack/budget/selection cell per job,
  using 8 targets and 5 victims.
- `defense/`: 62 jobs, one model/attack/budget/selection/defense cell per job,
  using 7 targets and 5 victims.
- `manifest.tsv`: source command, estimated L40S runtime, and requested walltime.

Every job requests one L40S and one CPU core. Most jobs request 7 GB of host
memory; a configuration with an unresolved memory-related termination requests
15 GB for its retry. Walltime is the configuration-specific L40S estimate plus
45 minutes. Standard
cells request at most 7 hours. Attack cells estimated to take more than 7 hours
are not capped: they request their full estimate plus 45 minutes and appear at
the end of `submit_attack.sh`. A timed-out cell can be submitted again and
resumes from its saved target/trial artifacts.

## Submission

From `/home/mmoslem3/scratch/attack_if`:

```bash
sh submit_attack.sh
sh submit_defense.sh
sh submit_attack-retry.sh
sh submit_defense_done.sh
```

Submit an individual configuration with, for example:

```bash
sbatch sbatch/attack/attack_001_convnet_gradmatch_dog_bird_b0_001_ours_std.sh
```

Wait for any attack jobs that create perturbations required by defense jobs
before submitting those defense jobs.

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
