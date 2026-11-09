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

Fir paired replacement, after the reported H100 timing of 9.3 seconds/epoch:
run `sh cancel_tiny_pending_fir.sh`, then `sh submit_tiny_pairs_fir_ghaddar.sh`.
Only pending jobs from this experiment's Fir ledger are canceled; running
jobs, including 3197751/3197752/3197753, are preserved. With those three kept,
the replacement has 168 jobs: nine surrogate groups, nine crafting groups,
and 150 victim groups, with at most two sequential tasks per job. All use
def-bghaddar, 1 CPU, 11 GB RAM and one H100. Every replacement job gets two
hours, including singleton jobs and resumed jobs.
The scientific plan, checkpoints, results and frozen code remain unchanged.
The paired runner uses a deadline shared by both sequential tasks, and
rerunning its submit script skips completed work and repairs dependencies.
Sync both new root scripts and `experiments/rebuttal_tiny_gm/`, including
`pair_fir.js` and `paired_job.sh`. The original Fir submit script delegates
to the paired submitter once `fir_pairs.json` has been created on Fir.

Current Fir downstream submission: `sh submit_tiny_finish_fir_ghaddar.sh`.
This leaves all surrogate jobs unchanged and schedules only crafting/victims:
six crafting jobs (three existing crafting tasks each) and 34 victim jobs
(nine independent 120-epoch trainings each, except the last job with three).
Total: 40 individual jobs, each 03:15:00, 12 GB, 1 CPU, 1 H100, def-bghaddar.
Every crafting job waits for all unfinished surrogate jobs; victim jobs wait
for the crafting jobs that produce their saved poisons. The original five
targets and five victims per cell, seeds, and scientific settings are unchanged.

The submitter uses the current Fir ledger, including the four replacement Lu
IDs. It prints the surrogate dependency list and refuses to duplicate older
active crafting/victim jobs. Rerun the same script to skip finished/active jobs,
resume partial groups, and repair dependencies after a stopped crafting job.
No new surrogate training or cancellation is performed.

New logs and scheduling records are outside the Git checkout:
`/scratch/mmoslem3/tiny_gm_vgg13_fir_runtime/`. This prevents repository sync
from replacing active logs. Copy the new root submit script and updated
`experiments/rebuttal_tiny_gm/` to Fir before running it. The main Fir submit
script delegates here after the downstream schedule is created.
