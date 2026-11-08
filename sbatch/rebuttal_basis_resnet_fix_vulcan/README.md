Corrected retries for ResNet20 GM 0.2%, lambda 0.1 and 10, replacing jobs
2182559 and 2182295. Both use `aip-bghaddar`, 2h, 6 GB RAM, one CPU and one L40S.

```sh
sh submit_basis_resnet_fix_vulcan_ghaddar.sh
```

The worker treats omitted `sel_mode` as disabled and loads the source run's
configured five surrogates. Crafting uses those five surrogates; the original
eight targets, five victims per target, 250 crafting steps, eight restarts and
50 victim epochs are preserved. No existing runtime snapshot is modified.
There were no completed poisons/victims to transfer from either affected job.

Results are isolated in `rebuttal_basis_resnet_fix_result/`. Rerunning this
submitter skips active/completed retry jobs. This submitter replaces only these
two cells; use it rather than the old batch submitters for their retries.

Mock regression check:

```sh
node experiments/rebuttal_basis_lambda/check_resnet_fix.js
```
