On Fir, from `/scratch/mmoslem3/PoisonBase`:

```sh
sh submit_basis_lambda_fir_ghaddar.sh
```

Copy this entire snapshot directory and the new root submitter to Fir. This is
exactly the former Killarney half: **33 jobs, 12 cells, 264 victim trainings**.
It covers all lambda-0.1 cells and Tiny ImageNet SAPA 0.5% at lambda 1.
It does not include the eight separate Tiny GM R/M jobs.

All jobs use `def-bghaddar`, `--gpus-per-node=h100:1`, one CPU, and unchanged
time/RAM requests: CIFAR-10 2h/6G, CIFAR-100 6h/7G, Tiny BP 4h/18G, and
remaining Tiny work 6h/18G. Crafting/victim dependencies and all target IDs,
surrogate choices, lambda values and training settings match the original plan.

Inputs use the original Vulcan-style paths on Fir, without `_new` suffixes:

```text
/scratch/mmoslem3/PoisonBase/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42/net_0.pt .. net_19.pt
/scratch/mmoslem3/PoisonBase/cache/surrogates/ResNet20BN_60ep_lr0.1_bs128_seed42/net_0.pt .. net_19.pt
/scratch/mmoslem3/PoisonBase/cache/surrogates/CIFAR100_ResNet18BN_60ep_lr0.1_bs128_seed42/net_0.pt .. net_19.pt
/scratch/mmoslem3/PoisonBase/cache/surrogates/TinyImageNet_ResNet18BN_60ep_lr0.1_bs128_seed42/net_0.pt .. net_19.pt
/scratch/mmoslem3/PoisonBase/data/cifar-10-batches-py/
/scratch/mmoslem3/PoisonBase/data/cifar-100-python/
/scratch/mmoslem3/PoisonBase/data/tinyimagenet.pt
```

The environment is `/home/mmoslem3/ENV/bin/activate`. The submitter checks input
sizes, Fir/H100 availability, and software modules before submitting. Each job
checks staged data and surrogate hashes before training.

Results and checkpoints are isolated in `rebuttal_basis_lambda_fir_result/`;
the existing Vulcan/Killarney snapshots and results remain untouched. Rerun the
same script to resume unfinished work, skip completed/active jobs, and repair
dependencies on replacement crafting jobs. Each cell summary combines its
target/victim splits only after its full grid finishes.

Lightweight local check (mock Slurm, no GPU work):

```sh
node experiments/rebuttal_basis_lambda/check_fir_submission.js
```
