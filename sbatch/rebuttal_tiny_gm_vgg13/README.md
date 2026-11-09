Only the last rebuttal table, `tab:reb-tiny-gm-extra-pairs`, is included.

VGG13BN, Tiny ImageNet, GM, epsilon 8/255, budgets 0.1% and 0.05%,
BASIS lambda=1 and one Rand draw per target. The three class pairs have six
fixed targets each. Methods and budgets share those targets and victim seeds.
Each of the twelve cells has six victims per target, giving 36 trials.

The shared 20-surrogate pool trains for 150 epochs without augmentation;
SGD learning rate 0.1 decays at epochs 90 and 120, batch size 128. Fourteen
surrogates use aip-bghaddar; six use aip-yiweilu. Each gets one 90-minute job.
Victims train for 120 epochs with crop/flip augmentation, learning-rate drops
at 60 and 90, batch size 128. Each victim gets one 60-minute Ghaddar job.
The VGG head uses the same adaptive final pooling as the timing benchmark.

There are 470 GPU tasks in 75 arrays: 20 surrogate, 18 crafting and 432 victim
tasks. Every task requests 1 CPU, 12 GB host memory and 1 L40S GPU. Crafting
uses five surrogates, eight restarts and 250 steps. Groups contain three
100-poison targets or six 50-poison targets. Every victim depends on its
crafting task; crafting depends on all surrogate tasks.

`sh submit_tiny_gm_vgg13_ghaddar.sh` submits or resumes the frozen plan.
Model, optimizer and CPU/CUDA/NumPy/Python RNG checkpoints are saved every
five training epochs. Crafting checkpoints are saved every 25 steps.
Completed tasks and active arrays are skipped. If only some elements of an
array have stopped, wait until that array finishes before rerunning the script.
Pending dependencies are updated when an upstream task is resubmitted.

Surrogate zero checks target labels, selection dispatch, the 64x64 model head,
and augmented SGD checkpoint equivalence before its full training. A successful
check is required before any crafting. Short GM graph checks and timings with
randomly initialized models are recorded separately under
`rebuttal_timing_result/tiny_5epochs_2222563/`; those poisons are never reused.

Results go to `rebuttal_tiny_gm_vgg13_result/`; `summaries/` reports a cell only
after all 36 victim results are present. Source and plan checksums are frozen
in `code.sha256`. Logs and the submission ledger are in this directory.

The same complete plan can also run on either remote cluster:
`sh submit_tiny_gm_vgg13_killarney_ghaddar.sh` from the Killarney checkout
(`/scratch/mmoslem3/attack_if`) uses only `aip-bghaddar`, `--gres=gpu:1`.
`sh submit_tiny_gm_vgg13_fir_ghaddar.sh` from the Fir checkout
(`/scratch/mmoslem3/PoisonBase`) uses only `def-bghaddar`, `--gpus-per-node=h100:1`.
Both keep all 470 tasks, targets, seeds, epochs, memory and time limits unchanged.
Each site has its own submission ledger, lock and log directory.

Copy the two submit scripts, `experiments/rebuttal_tiny_gm/submit.js`, and this
snapshot's `plan.json`, `job.sh`, `code.sha256` and complete `code/` directory.
The only required input excluded by Git is
`/scratch/mmoslem3/PoisonBase/data/tinyimagenet.pt` on Vulcan. Its destination is
`/scratch/mmoslem3/attack_if/data/tinyimagenet.pt` on Killarney or
`/scratch/mmoslem3/PoisonBase/data/tinyimagenet.pt` on Fir. If it already exists
at `/home/mmoslem3/scratch/data/tinyimagenet.pt`, the submitter links it into
the checkout instead. Both clusters use `/home/mmoslem3/ENV/bin/activate`.
No trained surrogates or saved poisons need copying for a fresh run.
