Only the last rebuttal table, `tab:reb-tiny-gm-extra-pairs`, is included.

VGG13BN, Tiny ImageNet, GM, epsilon 8/255, budgets 0.1% and 0.05%,
BASIS lambda=1 and one Rand draw per target. Each of the three pairs uses
five fixed targets, shared across methods and budgets. Each of the twelve
cells uses five victims per target, giving 25 trials per cell.

There are 338 individual Slurm jobs: 20 surrogate, 18 crafting and 300 victim
jobs. Each has its own job ID; no arrays are submitted. Every job requests
1 CPU, 12 GB RAM and one GPU. Surrogates train for 150 epochs with LR drops
at 90/120 and no augmentation; each gets 90 minutes. Victims train for 120
epochs with LR drops at 60/90 and crop/flip augmentation; each gets 60 minutes.
Crafting uses five surrogates, eight restarts and 250 steps, with up to three
100-poison targets or five 50-poison targets per 60-minute job.

Vulcan: `sh submit_tiny_gm_vgg13_ghaddar.sh` uses L40S GPUs, fourteen surrogate
jobs on aip-bghaddar and six on aip-yiweilu; all other jobs use aip-bghaddar.
Killarney: `sh submit_tiny_gm_vgg13_killarney_ghaddar.sh` uses only aip-bghaddar
with `--gres=gpu:1`, from `/scratch/mmoslem3/attack_if`.
Fir: `sh submit_tiny_gm_vgg13_fir_ghaddar.sh` uses only def-bghaddar
with `--gpus-per-node=h100:1`, from `/scratch/mmoslem3/PoisonBase`.

Each victim depends on its crafting job; crafting depends on all twenty
surrogates. Training and crafting checkpoint regularly. Rerun the same
submit script to resume stopped jobs; completed and active jobs are skipped,
and pending dependencies are repaired. Each cluster has a separate ledger
and log directory. Results use `rebuttal_tiny_gm_vgg13_5x5_result/`, separate
from the canceled six-target/six-victim plan. Summaries require all 25 trials.

Copy the submit scripts, `experiments/rebuttal_tiny_gm/submit.js`, and this
directory's `plan.json`, `job.sh`, `code.sha256` and complete `code/` directory.
The only input excluded by Git is Vulcan's
`/scratch/mmoslem3/PoisonBase/data/tinyimagenet.pt`; copy it to
`/scratch/mmoslem3/attack_if/data/tinyimagenet.pt` on Killarney or
`/scratch/mmoslem3/PoisonBase/data/tinyimagenet.pt` on Fir. An existing copy at
`/home/mmoslem3/scratch/data/tinyimagenet.pt` is linked into the checkout instead.
Both remote clusters use `/home/mmoslem3/ENV/bin/activate`.
