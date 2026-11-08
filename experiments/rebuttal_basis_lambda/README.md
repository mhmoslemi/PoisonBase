The 66 jobs are split into **33 on Vulcan and 33 on Killarney**. Each part
contains twelve complete cells (264 victim trainings), with no overlapping tasks
or dependencies between clusters.

On Vulcan:

```sh
sh submit_basis_lambda_vulcan_ghaddar.sh
```

On Killarney, from the checkout (normally `/scratch/mmoslem3/attack_if`):

```sh
sh submit_basis_lambda_killarney_ghaddar.sh
```

Vulcan runs all lambda-10 cells and Tiny GM 0.5% at lambda 1. Killarney runs
all lambda-0.1 cells and Tiny SAPA 0.5% at lambda 1. Each crafting job stays on
the same cluster as its dependent victim jobs. Both scripts use the shared
snapshot under `sbatch/rebuttal_basis_lambda_killarney/`; keep that directory
when syncing either script. Per-cluster submission ledgers and log folders
prevent synchronized job IDs from being mistaken for local jobs.

This prepares only the 24 missing BASIS cells in `tab:reb-c-scaling`:
lambda 0.1 and 10 for all eleven rows, and lambda 1 for Tiny ImageNet GM/SAPA
at 0.5%. Existing lambda-1 cells and all R/M runs are left alone.

| Work | Jobs | Wall time | RAM |
| --- | ---: | ---: | ---: |
| CIFAR-10: one cell | 6 | 2 h | 6 GB |
| CIFAR-100: one cell | 8 | 6 h | 7 GB |
| Tiny BP 0.2%: one target, four victims | 8 | 4 h | 18 GB |
| Tiny GM 0.2%: one target, four victims | 8 | 6 h | 18 GB |
| Tiny GM/SAPA 0.5%: one target's crafting | 24 | 6 h | 18 GB |
| Tiny GM/SAPA 0.5%: two targets, four victims each | 12 | 6 h | 18 GB |

Total: **66 jobs, 120 poison sets, 528 victim trainings**. Every job uses
`aip-bghaddar` and one CPU. Vulcan requests `--gres=gpu:l40s:1`; Killarney
retains its established `--gres=gpu:1` request.
CIFAR-10 uses the same eight targets and victim IDs 0--4 as the row's copied
BASIS result. CIFAR-100 and Tiny ImageNet use the original four targets and
victim IDs 0--3. Each victim trains for 50 epochs. The two new Tiny 0.5% rows
reuse the four targets and training configuration of the corresponding 0.2% run.

The frozen plan records the original source logs, their hashes, exact target
IDs, and all training/crafting arguments. Only lambda, the requested budget,
and the evaluation counts change. Selection minimizes the original
`z(cosine distance) + lambda*z(logit margin)` score over 20 surrogates;
crafting uses the same first five surrogates, 250 steps and eight restarts
for GM/SAPA. SAPA retains `sharp_mode=worst`, sigma 0.05, and 20 samples.
There is no surrogate training.

The old ConvNet lambda sweep is not reused: those runs used 750 crafting
steps and six targets, whereas these rows require 250 steps and eight targets.
The legacy ResNet18 network factory's clock-based initialization is retained;
actual effective seeds are recorded, as in the existing M/R runner.

Splits use the saved Killarney timings: approximately 4,773 seconds to craft
200 Tiny poisons and 2,352 seconds per Tiny victim. Scaling crafting to 500
poisons gives roughly 3.3 hours; adding four victims would leave insufficient
margin under six hours. Crafting therefore has its own job at 0.5%; each
dependent victim job performs eight trainings (about 5.2 hours plus setup).
These are estimates; interrupted work can be resumed.

Results persist in `<checkout>/rebuttal_basis_lambda_result/`; each cell's
summary combines every split and reports ASR only when its complete grid exists.
Rerun the same submitter after a pause: completed and active jobs are skipped,
and dependencies on replacement crafting jobs are repaired. Crafting checkpoints
every 25 steps; victim checkpoints every five epochs and on a stop signal.
BP's short crafting stage restarts if interrupted before saving its poison.
Logs print completed epochs on resumption and progress every epoch.

The launcher checks input existence/sizes before submitting. Every allocated
job validates code and staged data/surrogate hashes before training. It uses
the existing Killarney environment `/home/mmoslem3/ENV/bin/activate` and data
under `<checkout>/data` (fallback `/home/mmoslem3/scratch/data`). No extra
preflight job or changes to old snapshots are needed.

Required trained pools on Killarney, each containing `net_0.pt` through
`net_19.pt`:

```text
/scratch/mmoslem3/attack_if/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42/
/scratch/mmoslem3/attack_if/cache/surrogates/ResNet20BN_60ep_lr0.1_bs128_seed42/
/scratch/mmoslem3/attack_if/cache/surrogates/CIFAR100_ResNet18BN_60ep_lr0.1_bs128_seed42_new/
/scratch/mmoslem3/attack_if/cache/surrogates/TinyImageNet_ResNet18BN_60ep_lr0.1_bs128_seed42_new/
```

The last two retain the previously requested `_new` destinations on Killarney.
On Vulcan, all pools are read from their original names without `_new`.
Exact hashes
are in `sbatch/rebuttal_basis_lambda_killarney/surrogates.tsv`; data paths and
hashes are in `data.tsv`. Remote checkpoint availability has not been verified.

Lightweight checks (mocked Slurm, no GPU work or Python on login):

```sh
bash -n submit_basis_lambda_vulcan_ghaddar.sh submit_basis_lambda_killarney_ghaddar.sh sbatch/rebuttal_basis_lambda_killarney/job.sh
node experiments/rebuttal_basis_lambda/check_submission.js
```
