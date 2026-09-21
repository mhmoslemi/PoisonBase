# Same 40 scoring experiments on attack_if

Submit all jobs with one command:

```bash
bash /home/mmoslem3/scratch/attack_if/submit_score_alternatives_40_attack_if.sh
```

The submitter validates the 40 individual jobs, creates the log directory, and
submits them all. Every job activates the environment itself.

- Project: `/home/mmoslem3/scratch/attack_if`
- Data: `/home/mmoslem3/scratch/data`, matching the previous attack_if jobs
- Environment: `/home/mmoslem3/ENV/bin/activate`
- Account: `aip-boyuwang`
- Per job: one L40S, 7 GB host RAM, `03:20:00`, one experiment, 8 targets x 6 victims
- Results: `/home/mmoslem3/scratch/attack_if/score_alternatives_8x6_20260921_result/cell_<ID>/`

`manifest.tsv` is identical to the original 40-job batch: five new scoring
methods over the same eight ConvNet attack/budget settings. Existing baseline
methods are not resubmitted. Targets, model pools, crafting, victim training,
completion checks, and resume behavior all use the shared implementation in
`../score_alternatives_8x6_20260921/`; only the filesystem paths and job names
change. The shared batch directory and updated `final_update.py` must be present
in the attack_if checkout, along with its existing `networks.py` and `utils.py`.

For a submission preview, prefix the same command with `DRY_RUN=1`.
