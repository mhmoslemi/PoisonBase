# Retry remaining rebuttal work

Run `sh submit_unfinished_rebuttal.sh` from the repository.

The manifest includes exactly 33 unfinished tasks from the checked logs:

- Seven first-table GM BASIS tasks (original W1 indices 3--8 and 10), requesting
  2 hours, 1 CPU, 6G and one L40S. Conditions 1--6 need 66--68 additional
  victims each; full-access condition 8 needs 42. Completed victims and poison
  caches are reused. An interrupted W1 victim restarts its 50-epoch training;
  other completed victims remain untouched.
- Twenty-four ResNet20 ASD bundles (indices 8--31), requesting 3 hours, 1 CPU,
  6G and one L40S. Each has only 2--3 victims unfinished; a saved checkpoint
  already contains part of the first unfinished victim.
- Two ResNet20 Telltale bundles (indices 34--35), requesting 3 hours, 1 CPU,
  16G and one L40S. They have 32 and 28 victims unfinished, respectively;
  completed detection masks and training checkpoints are reused.

All jobs use `aip-bghaddar`. No perturbation dependency remains: job 2168638
completed and its four saved base/delta pairs are present. The retry worker
uses the existing frozen defense code, changes only its self-pause deadline
to 2h55m, and preserves epoch/AE/clustering resume. The 24 ASD and two Telltale
jobs ended with exit 75 after saving progress at the original six-hour bound;
these are intentional resumable stops, not algorithm errors or OOMs.

The script skips completion markers and the latest active job for each task,
and appends job IDs to both the original ledgers and this retry ledger. Re-run
the same script if necessary. Logs go to `sbatch/rebuttal_resume/logs`.
Creating and checking this script did not submit any real jobs.

`defense_cells.json` records the per-cell audit used to update the TeX. Only
the exact completed target/victim grids were filled. Expanded W1 GM BASIS
results remain pending; their older 5x4 values are preserved as TeX comments
so the visible first table has one consistent 7x8 protocol.
