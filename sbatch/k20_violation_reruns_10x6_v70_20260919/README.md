# K=20 violation reruns (full 10x6 protocol)

This batch contains only the 18 current `ensemble.tex` cells for which the
reported `K=20` ASR is below at least one of `K=1`, `K=3`, or `K=10`.

Every Slurm script is one configuration and pins:

- `K=20` (never a lower K)
- 10 target images and 6 victim seeds, for 60 evaluations
- `base=ours`, `base_dist=cosine_norm`, `lambda_margin=1`
- Jacobian scoring off and 250 poison-optimization steps
- victim training for 70 epochs with decay at epoch 50
- selection architecture `S` as listed in the manifest
- victim-matched poison crafting with the same first five `V` checkpoints

Outputs use the new directory
`$ROOT/k20_violation_10x6_v70_20260919_result`, so old partial results cannot be
mistaken for completion. The common runner validates exactly 60 unique
`(target, victim)` result rows before accepting a job as complete.

Preview without submitting:

```bash
DRY_RUN=1 bash sbatch/k20_violation_reruns_10x6_v70_20260919/submit_all.sh
```

Submit:

```bash
bash sbatch/k20_violation_reruns_10x6_v70_20260919/submit_all.sh
```
