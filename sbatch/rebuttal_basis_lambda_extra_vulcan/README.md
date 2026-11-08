The 33 BASIS jobs formerly assigned to Killarney/Fir, now on Vulcan:

```sh
sh submit_basis_lambda_remaining_vulcan.sh
```

Exactly **20 jobs on `aip-bghaddar` and 13 on `aip-boyuwang`**, one CPU and
one L40S each, with unchanged 2/4/6-hour limits and 6/7/18 GB RAM. All target
IDs, victim counts, selection/crafting surrogates, attacks and lambda values
match the Fir plan: 12 cells and 264 victim trainings. No existing experiment
or table values are edited.

Ghaddar handles the twenty Tiny ImageNet lambda-0.1 jobs. Wang handles the
seven CIFAR jobs and the six Tiny SAPA lambda-1 jobs. Every dependency stays
within its account. This is separate from the original active Vulcan half.

Results/checkpoints go to `rebuttal_basis_lambda_extra_result/`. Rerun this
script to resume unfinished jobs, skip completed/active tasks, and repair
dependencies after replacing crafting jobs. The launcher also rejects active
jobs with matching task names outside its ledger to avoid duplicate submission.
Inputs use original Vulcan surrogate directories without `_new` suffixes.

Lightweight mocked check:

```sh
node experiments/rebuttal_basis_lambda/check_remaining_vulcan_submission.js
```
