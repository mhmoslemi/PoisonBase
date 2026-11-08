On Vulcan:

```sh
sh submit_meta_k2_vulcan_lu.sh
```

Six independent jobs, each `aip-yiweilu`, 3 hours, one CPU, 6 GB RAM,
one full L40S. Cells: ConvNet BP 0.5%, ConvNet SAPA 0.2%, ResNet20 BP 0.2%,
ResNet20 GM 0.2%/0.5%, and ResNet20 SAPA 0.5%.

This is a separate **two-selection-surrogate ablation**, using the saved
per-surrogate Meta-Grad weights from surrogate IDs 0 and 1. Their old protocol
identity and input hashes are checked; their normalized weights are averaged,
then the top 100/250 dog candidates are selected. No new score computation or
surrogate training is needed. Both crafting and victim training must be rerun
because the selected bases can change.

Crafting retains the original first five surrogates, 250 steps, GM/SAPA eight
restarts, BP one restart, epsilon 8/255, and unchanged augmentation/SAPA settings.
Every cell uses its original eight pinned targets and victim IDs 0--4,
trained for 50 epochs: **240 victim trainings** total. Only five surrogate
checkpoints are loaded; the selection ranking itself uses exactly two cached
surrogate scores.

The saved 8x6 runs took approximately 41/50/78/109/140/139 minutes for these
six cells. At 8x5, estimates are about 34/44/65/96/127/126 minutes plus setup,
so each receives a three-hour limit. This is based on L40S logs, not a new GPU
measurement.

The launcher verifies required files/sizes before submitting. Each compute job
checks complete input SHA256 hashes before running. It uses the existing
`/home/mmoslem3/ENV/bin/activate` environment. Results and cached poisons go to
`rebuttal_meta_k2_result/`, leaving the original table and 20-surrogate results
unchanged. Rerun the same submitter to skip completed or active jobs and retry
unfinished ones. Completed poison sets and victim trials are reused; an
interrupted victim or unfinished crafting stage restarts.

Lightweight submission check (mock Slurm only):

```sh
bash -n submit_meta_k2_vulcan_lu.sh sbatch/rebuttal_meta_k2_vulcan/job.sh
node experiments/rebuttal_meta_k2/check_submission.js
```
