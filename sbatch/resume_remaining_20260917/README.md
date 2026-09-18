# Remaining-run recovery jobs

This directory submits only the incomplete configurations found after commit
`988bebeb`:

- 16 fresh architecture-transfer configurations;
- 6 static Gao selector configurations; and
- 2 FUS configurations, with only their incomplete target partitions submitted.

Every submission resumes its own saved `results*.csv` and `poison_cache`
directory. It does not rerun completed target/victim pairs. A configuration that
has reached 60/60 by submission time is detected and skipped automatically.

The transfer and static-selector jobs retain their 2:30 wall time. FUS retains
the original 4-hour per-part limit; currently only partitions 0 and 1 require
work. FUS merge jobs request 10 minutes and run after the resumed parts succeed.

Submit everything from the PoisonBase root with:

```bash
bash sbatch/resume_remaining_20260917/submit_all.sh
```

Or submit the two groups separately:

```bash
bash sbatch/resume_remaining_20260917/submit_transfer.sh
bash sbatch/resume_remaining_20260917/submit_selectors.sh
```

The scripts use `/home/mmoslem3/scratch/PoisonBase`, its `data` directory, and
`/home/mmoslem3/ENV/bin/activate` by default. Existing account assignments are
preserved. Old merge jobs `970836` and `970844` can be cancelled separately if
they are still waiting on their original failed dependencies:

```bash
scancel 970836 970844
```

To inspect exactly what would be submitted without calling `sbatch`:

```bash
DRY_RUN=1 bash sbatch/resume_remaining_20260917/submit_all.sh
```
