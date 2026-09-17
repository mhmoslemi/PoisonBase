# Fresh crafting-to-victim architecture transfer

This directory contains 72 one-configuration Slurm jobs:

- attacks: GM and SAPA;
- budgets: 0.002 and 0.005;
- selector ensemble sizes: K=10 and K=20;
- selection/crafting architecture: ConvNetBN, ResNet20BN, or VGG13BN, with S=A;
- victim architecture: ConvNetBN, ResNet20BN, or VGG13BN;
- 10 fixed dog-to-bird targets and 6 victim seeds per job.

Every job passes `--FORCE`, so it selects bases and crafts perturbations again.
It does not import poison caches from any earlier experiment.  Seed-determined
surrogate and clean-victim network checkpoints are still shared.

The result root is
`/home/mmoslem3/scratch/PoisonBase/victim_transfer_fresh_20260917_result`.

Submit all jobs from the repository root with:

```bash
bash sbatch/victim_transfer_fresh_20260917/submit_all.sh
```
