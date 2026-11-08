Only the twelve empty M-only/R-only cells in reviewer 2 W1 (`tab:reb-r-vs-m`).
No existing experiments, results, or TeX are changed. No surrogate training is performed.

Run on Killarney from the checkout (normally `~/scratch/attack_if`):

```sh
sh submit_mr_killarney_ghaddar.sh
```

All 29 jobs use `aip-bghaddar`, one CPU, and the requested `--gres=gpu:1`:

| Work | Jobs | Time per job | RAM |
| --- | ---: | ---: | ---: |
| Preflight | 1 | 40 min | 18 GB |
| CIFAR-10: one complete M/R cell per job | 4 | 2 h | 6 GB |
| CIFAR-100: one complete M/R cell per job | 4 | 3 h | 7 GB |
| Tiny ImageNet BP: one target + four victims | 8 | 4 h | 18 GB |
| Tiny ImageNet GM: two targets' crafting | 4 | 4 h | 18 GB |
| Tiny ImageNet GM: one target's four victims | 8 | 4 h | 18 GB |

The 28 experiment jobs cover 52 poison sets and 228 victim trainings. Every victim
trains for 50 epochs. CIFAR-10 has five targets × five victims per cell;
CIFAR-100 and Tiny ImageNet have four targets × four victims. Target IDs, victim
nominal seeds, trained surrogate pools, and attack/training parameters come from the
corresponding BASIS source logs recorded in `sbatch/rebuttal_mr_killarney/plan.json`.
The harder CIFAR-10 cells retain DPP alpha=2. Only the score's M/R coefficient changes:
M-only uses standardized margin; R-only uses standardized cosine distance, computed
per surrogate before averaging, as in the two terms of the BASIS selector.
The legacy `utils.get_network` overrides the seed with the clock for ResNet18BN.
That existing behavior is preserved, with each trial's `effective_seed` recorded;
the historical ResNet runs cannot be claimed to have identical initial weights.

The launcher's file/size checks run before submission. The GPU preflight verifies
data and surrogate SHA256 hashes, checkpoint compatibility, target labels,
selection formulas/counts, original versus checkpointed SGD, GM crafting resume,
and full poison-batch Tiny GM/BP graphs under 18 GB RAM. It does not generate
experiment results. All experiment jobs depend on successful preflight; Tiny GM
evaluation also depends on the corresponding crafting job. A failed preflight
therefore prevents experiment jobs from starting. Preflight is where Killarney
software/GPU compatibility and the requested 18 GB are checked, not an assertion
that those checks have already passed.

The wrapper stages input files to `$SLURM_TMPDIR`; results and checkpoints persist
under `<checkout>/rebuttal_mr_result`. GM saves every 25 crafting steps, victims
every five epochs and on a stop request. BP's short crafting step (~81 seconds)
restarts only if interrupted before its completed poison file is saved. Logs show
saved epochs on resume and progress every epoch. Rerun the same shell script after
timeouts: it skips completed/active jobs, resumes checkpoints, and repairs pending
dependencies when an upstream job is replaced.

Per-cell JSON/CSV files in `rebuttal_mr_result/summaries` combine all target/victim
splits. ASR/CTA are reported only after every expected trial has completed; partial
summaries show finished/expected counts. Nothing edits the rebuttal automatically.

Killarney needs the matching three 20-model surrogate pools and all three datasets.
The exact 71 input paths on Vulcan are in
`sbatch/rebuttal_mr_killarney/copy_from_vulcan.txt`. On Killarney, the CIFAR-100 and
Tiny ImageNet pool directories must have `_new` appended to their names:

```text
/scratch/mmoslem3/attack_if/cache/surrogates/CIFAR100_ResNet18BN_60ep_lr0.1_bs128_seed42_new/
/scratch/mmoslem3/attack_if/cache/surrogates/TinyImageNet_ResNet18BN_60ep_lr0.1_bs128_seed42_new/
```

Each contains `net_0.pt` through `net_19.pt`, copied from the corresponding Vulcan
directory without the `_new` suffix. The submitter and staging job require these
new directories; they do not fall back to the old ResNet pools. The ConvNet pool
keeps its original directory. Staging retains the logical cache names expected by
the unchanged worker and checksum manifest. Data defaults to `<checkout>/data`, falling back
to `/home/mmoslem3/scratch/data`; `DATA_ROOT` and `ENV_ACTIVATE` can override those
locations. The default environment remains `/home/mmoslem3/ENV/bin/activate`.

Lightweight submission checks (no Slurm submission or Python on login):

```sh
bash -n submit_mr_killarney_ghaddar.sh sbatch/rebuttal_mr_killarney/job.sh
node experiments/rebuttal_mr/check_submission.js
```
