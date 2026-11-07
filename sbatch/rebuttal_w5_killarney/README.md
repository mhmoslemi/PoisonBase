# W5 on Killarney: 42 three-hour Ghaddar jobs

From the Killarney checkout, normally `~/scratch/attack_if`:

```sh
sh submit_w5_killarney_ghaddar.sh
```

All jobs request `aip-bghaddar`, `--gres=gpu:1`, 1 CPU, 6G, and 03:00:00,
as requested for Killarney. No partition is specified. The script's location
determines the checkout; it does not use Vulcan's hardcoded project directory.
The existing environment defaults to `/home/mmoslem3/ENV/bin/activate`.
CIFAR-10 is read from `<checkout>/data`, falling back to
`/home/mmoslem3/scratch/data`, exactly as the earlier Killarney launcher.
`ENV_ACTIVATE`, `DATA_ROOT`, and `SOURCE_ROOT` can override those existing paths.

| Defense | ConvNet jobs | ResNet20 jobs | Total |
|---|---:|---:|---:|
| Deep k-NN | 2 | 4 | 6 |
| Activation Clustering | 2 | 4 | 6 |
| PGD adversarial training | 8 | 22 | 30 |
| Total | 12 | 30 | 42 |

These are exactly the previously agreed 1,446 final defended victims and 90
poisoned datasets. There are no additional experiments. Each bundle has at
most 150 minutes of estimated work, leaving 30 minutes within its walltime.
AT budgets 360/630 seconds per ConvNet/ResNet victim, measured on L40S.
Filtering victims conservatively budget 55/110 seconds, since retained sample
counts vary. Bundles can cross table cells and split a target's victim seeds.
`plan.json` records every target, victim, and frozen source configuration.

## Files Git does not transfer automatically

All 458 required saved input files were present on Vulcan when preparing this
batch: 2,555,107,368 bytes (2.38 GiB), excluding CIFAR-10. They include:

- The saved poison deltas and matching base-index files/bundles under
  `ours_result/<the 12 source runs>/`.
- All 90 poison identity records under
  `rebuttal_saved_defenses_result/poison_records/`.
- For each of the 90 initial poisoned-data feature models, its
  `detector/trajectory/checkpoint.pt`, `detector/protocol.json`, and
  `detector/detection.npz` under `rebuttal_saved_defenses_result/telltale/`.
- CIFAR-10's seven batch/metadata files, at either supported data location.

`.pt` weights/deltas and `data/` are ignored by Git; legacy `bases.json` files
are also ignored. Other JSON/NPZ metadata may be untracked. The exact lists
below include every required input regardless of Git status.

Some deltas reside in Vulcan's `/scratch/mmoslem3/attack_if`, while metadata
and other inputs reside in `/scratch/mmoslem3/PoisonBase`. Merge both sets into
the same Killarney checkout. After copying this launcher and its entire
`sbatch/rebuttal_w5_killarney/` directory, run on Killarney:

```sh
cd ~/scratch/attack_if
# Replace VULCAN_HOST with your normal SSH destination for Vulcan.
rsync -av --files-from=sbatch/rebuttal_w5_killarney/vulcan_poisonbase_files.txt VULCAN_HOST:/scratch/mmoslem3/PoisonBase/ ./
rsync -av --files-from=sbatch/rebuttal_w5_killarney/vulcan_attack_if_files.txt VULCAN_HOST:/scratch/mmoslem3/attack_if/ ./
sh submit_w5_killarney_ghaddar.sh
```

`artifacts.tsv` lists destination paths, expected byte counts, and roles;
`vulcan_sources.tsv` maps every source path to its Killarney destination.
Preflight aborts before submission if any required file is absent/incomplete.
The existing environment must include torch, torchvision, numpy, scipy, sklearn.
The Vulcan timing environment was torch 2.9.1, numpy 2.3.3, sklearn 1.8.0;
each real job logs its actual versions. No clean victim/attacker-surrogate
checkpoint pools or ASD final models are required by these jobs.

## Runtime and resume

The self-contained `code/` snapshot includes W5, its saved-poison loader,
the model/dataset helpers, and licensed reference functions. Its checksums
and the plan are validated before submission and again at job start.
All scientific settings remain unchanged from the implementation checks:
k=5,000, ART PCA(10)/KMeans(2)/smaller-cluster defaults, PGD-10 at 8/255
with 2/255 steps, and 50-epoch final victims. The aggressive ConvNet filter
settings and feature-model adaptation remain as documented in the W5 source
README; this launcher does not retune them.

Results and epoch/RNG/optimizer checkpoints are saved under
`<checkout>/rebuttal_w5_result`. Logs are in this batch's `logs/` directory.
The same command skips completed bundles and active queued/running jobs,
and resubmits unfinished ones. No dependencies or cancellations are needed:
the inputs already exist, and shared feature/filter files are locked.
At five minutes before the walltime, the deadline requests a checkpointed
stop; Slurm also signals the worker three minutes before expiry. Exit 75
means saved progress, ready to resume with the same submit command.

No real jobs were submitted during preparation. Shell syntax and mocked
submission/retry checks were performed on Vulcan; Killarney execution has
not been tested. The two supplied interactive allocations had expired
before this submission-preparation turn, so no new GPU check was launched.
