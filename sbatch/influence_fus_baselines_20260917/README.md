# Gao difficulty selectors and adapted Xia FUS

The directory contains 32 matched-architecture configurations:

- selectors: Gao loss, exact full gradient norm, forgetting events, and adapted FUS;
- attacks: GM and SAPA;
- poison budgets: 0.002 and 0.005;
- models: ConvNetBN and ResNet20BN;
- selector ensemble: K=20;
- dog-to-bird, 10 pinned targets, 6 victim seeds, and 250 final crafting steps.

The Gao definitions follow the authors' released code: loss and full gradient
norm are measured after epoch 10; forgetting counts correctness transitions
from correct to incorrect; never-learned samples receive the maximum score.
The two `precompute_metrics_*.sh` array jobs train the 20 trajectory seeds once,
and the corresponding experiment jobs depend on successful completion.

FUS is explicitly an adaptation to targeted clean-label poisoning. At every
update it crafts the current GM/SAPA poisons, trains a search model, keeps the
half with the most forgetting events, and randomly replenishes the rest. It uses
10 update rounds, alpha=0.5, and 50-epoch search models. Each of the eight FUS
configurations is a four-task Slurm array: the pinned targets are split 3/3/2/2,
and every target still runs all six victim seeds. Each part requests four hours.
A small dependent CPU job validates and merges the four parts into the canonical
10-target, 60-evaluation result directory. The 24 static Gao jobs each request
2 hours 30 minutes.

Nothing is submitted by the generator. To submit preprocessing and all results:

```bash
bash sbatch/influence_fus_baselines_20260917/submit_all.sh
```

This submits 24 single Gao experiment jobs and eight four-element FUS arrays,
for 56 experiment GPU tasks total. It also submits eight merge jobs that start
only after all four parts of their respective FUS array succeed.

Results are written under:

```text
/home/mmoslem3/scratch/PoisonBase/influence_fus_baselines_result
```

## Compute and memory accounting

Each result directory also saves method-specific overhead instead of mixing it
with victim training:

- `overhead/target_<id>.json` records selector wall time, final-crafting wall
  time, absolute and incremental PyTorch CUDA peaks, and process peak RSS for
  that target;
- `overhead/summary.json` aggregates the available target records (totals,
  per-target means, and maximum memory); the main `summary.json` repeats the
  scalar aggregate fields;
- `job_gnu_time_<jobid>_<task>.txt` is `/usr/bin/time -v` output for the complete
  Python experiment, including victim evaluation. For FUS these files remain in
  `parts/part_<n>_of_4/`, and a new file is retained for every resumed attempt.

For Gao selectors, the one-time preprocessing cost is stored separately under
`cache/selector_metrics/.../class5/`: every `net_<id>.resources.json` records
training time, metric-evaluation time, CUDA peak memory, and CPU peak RSS for
one of the 20 surrogate shards. `net_<id>.gnu_time_<jobid>.txt` contains the
corresponding whole-process measurement. This shared preprocessing should be
reported once per architecture, not multiplied by the number of attacks or
budgets. For adapted FUS, the repeated proxy crafting and search-model training
are included directly in each target's selector phase.

CUDA byte fields come from PyTorch's allocator. `peak_allocated` is the absolute
peak including models already resident on the device; `incremental_peak` is the
additional peak above the live allocation at phase start. RSS fields are KiB.
