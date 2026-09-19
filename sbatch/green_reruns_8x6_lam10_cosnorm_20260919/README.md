# Green-highlighted reruns

This batch contains the 55 green-highlighted cells from the supplied table
image. The partially highlighted BP, rho=.005, S=VGG13, V=VGG13, K=3 cell is
included.

Each script represents one table cell and pins:

- 8 target images and 6 victim seeds (48 evaluations)
- the cell's highlighted selection-ensemble size K
- `base=ours`, `base_dist=cosine_norm`, and `lambda_margin=10`
- Jacobian scoring off
- 250 poison-optimization steps
- the original 50-epoch victim schedule with decay at epoch 40
- victim-matched poison crafting using the same first five V checkpoints

Results are isolated in
`$ROOT/green_rerun_8x6_lam10_cosnorm_20260919_result`. Completion requires 48
unique `(target, victim)` rows.

Preview:

```bash
DRY_RUN=1 bash sbatch/green_reruns_8x6_lam10_cosnorm_20260919/submit_all.sh
```

Submit:

```bash
bash sbatch/green_reruns_8x6_lam10_cosnorm_20260919/submit_all.sh
```
