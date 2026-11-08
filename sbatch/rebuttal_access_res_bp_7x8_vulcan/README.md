ResNet20 BP columns only for `tab:reb-access` in `Rebutal/Rebutal.tex`.

Run on Vulcan:

```sh
sh submit_w1_resnet_bp_7x8_vulcan.sh
```

41 jobs: six Rand jobs and 35 BASIS jobs. Each requests `aip-bghaddar`, four hours,
one CPU, 6 GB RAM, and one L40S. A job covers at most two draws within one cell.
Historical ResNet20 BP logs show 95–97 seconds per 50-epoch victim: two draws mean
112 victims, about three hours plus crafting/setup. Total work is 4,032 victims
and 504 poison sets, approximately 108–115 GPU-hours.

The seven BASIS cells retain the existing condition IDs and seed formulas:

| Condition | Surrogate data | Candidate pool |
| --- | ---: | ---: |
| 8 | 100% | 100% |
| 1 | 50% | 100% |
| 2 | 20% | 100% |
| 3 | 100% | 50% |
| 5 | 50% | 50% |
| 4 | 100% | 20% |
| 6 | 20% | 20% |

Each BASIS cell uses three surrogate draws × three candidate draws × seven
targets × eight victims = 504 trials. Rand uses three candidate draws × seven
targets × eight victims = 168 trials, reported once per candidate fraction
(conditions 1/3/4 for 100%/50%/20%). Rand always crafts on the full-data surrogate
pool; BASIS selects and crafts with the corresponding full/partial pool.

The runner is a frozen copy of the code that produced the existing ConvNet
columns, with model/output location/targets changed for ResNet20 BP. All poison
selection, candidate-subset construction, score normalization, crafting, and SGD
settings remain the same: rho=0.002, epsilon=8/255, BP sample mode, 250 craft
steps, alpha=1/255, one BP restart, first five crafting surrogates, and 50 victim
epochs with SGD lr=0.1, momentum=0.9, batch=125, decay at epoch 40, no weight decay
or victim augmentation. Every victim uses the full poisoned training set.

Targets are the first seven pinned ResNet20 BP targets from the paper's target
file: 3495, 2903, 6697, 2315, 6899, 2312, 6353. Victim IDs are 0–7, with seed
4200000 + target*100 + victim. Candidate draws use the same stratified subsets
and seeds as ConvNet. The full-data surrogate draws retain the existing windows
0–19, 5–24, and 10–29; partial draws use the matching three trained 20-model pools
at each of 20% and 50% training data. These draws overlap for the full-data case,
as in the existing ConvNet experiment. No surrogates or clean controls are trained.

The submitter verifies all 150 required surrogate checkpoints, six partial-pool
protocols, pinned targets, code hashes, and dataset files before submitting.
Results persist under `rebuttal_access_res_bp_7x8_result/bp`; only input data and
temporary caches are staged under `$SLURM_TMPDIR`. Rerunning skips completed or
active tasks and resubmits unfinished tasks. Completed poisons and victims are
reused. An interrupted victim restarts that one 50-epoch training, as in the
existing access experiment; completed victim JSONs are retained.

Each shard writes `summary_7x8_draws_START_STOP.json`. Once all draws are present,
the finishing job writes `summary_7x8.json` with 168 or 504 trials for the cell.
Summaries are also printed to the job log. Nothing fills the new TeX cells until
complete results exist.
