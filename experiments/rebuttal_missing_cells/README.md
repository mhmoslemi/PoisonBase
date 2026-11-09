Only the ten previously unsubmitted cells in `tab:reb-c-scaling`:
CIFAR-100 sea→willow GM R/M; wardrobe→lawn mower SAPA R/M;
Tiny ImageNet GM/SAPA at 0.5%, each with Rand/R/M.

All use four original targets, four victims per target, 50 victim epochs,
five crafting surrogates, 250 crafting steps and eight restarts. R/M use
the original twenty selection surrogates. Rand uses the historical fixed
CPU-generator seed `42*100003 + target`. Source configuration is pinned in
the plan; existing batches and results are untouched.

`sh submit_missing_cells_vulcan.sh` submits or resumes 52 jobs:
four CIFAR-100 whole-cell jobs (2h, 7G, `aip-yiweilu`), plus 24 Tiny crafting
jobs (5h, 18G) and 24 dependent Tiny victim jobs (3h, 18G), all Tiny on
`aip-bghaddar`. Every job uses one CPU and one L40S.

Snapshot: `sbatch/rebuttal_missing_cells_vulcan/`.
Results: `rebuttal_missing_cells_result/`; complete-cell summaries combine splits.
The existing atomic crafting/SGD checkpoints, poison identity checks, and
active/completed-job skips remain in use. Resubmission repairs dependent jobs.

`node experiments/rebuttal_missing_cells/check_submission.js` mocks Slurm.
Selection checks run inside each allocation, never on the login node.
