# Five new scoring alternatives: 40 ConvNet experiments

Only the five additional methods in the first screenshot are submitted. The
existing RAND, R, M, full-alignment, and BASIS results are retained.

| Attack in table | Runner | Poison ratios | New selectors | Jobs |
|---|---|---|---|---|
| BP | `fc --fc_mode sample` | 0.001, 0.002 (0.1%, 0.2%) | all five | 10 |
| GM | `gradmatch` | 0.001, 0.002, 0.005 | all five | 15 |
| SAPA | `sapa` | 0.001, 0.002, 0.005 | all five | 15 |

Every `job_*.sh` runs exactly one attack/budget/selector experiment with eight
targets and six victims per target, for 48 evaluations. All 40 jobs use
`aip-boyuwang`, one L40S, 7 GB host RAM, and `03:20:00` walltime.

## Submission

Sync the updated `final_update.py`, this entire batch directory, and the root
`submit_score_alternatives_40.sh` into `/home/mmoslem3/scratch/PoisonBase`.
The existing `networks.py` and `utils.py` must also be present.

```bash
cd /home/mmoslem3/scratch/PoisonBase
source /home/mmoslem3/ENV/bin/activate
DRY_RUN=1 bash submit_score_alternatives_40.sh  # preview all 40
bash submit_score_alternatives_40.sh          # submit all 40
```

Individual experiments can be submitted directly with `sbatch job_path`.
Create `sbatch/logs/` first if submitting directly. `manifest.tsv` maps every
job to its experiment. No baseline jobs are included.

## Protocol pinned from the table's original experiments

- CIFAR-10 from `/home/mmoslem3/scratch/PoisonBase/data`.
- ConvNetBN, `dog-bird`: dog poison/adversarial class; bird target class.
- Seed 42; selection uses 20 existing 60-epoch surrogates; crafting uses the
  first five of those same checkpoints.
- Perturbation bound `0.0313725` (the original approximately 8/255 setting),
  250 crafting steps, step size `0.0039216`, eight GM/SAPA restarts, and one
  FC restart. BP uses the repository's original `fc_mode=sample` mapping.
- Default differentiable augmentation during GM/SAPA crafting; SAPA uses
  `sharp_mode=worst`, `sharp_sigma=0.05`.
- Victims: 50 epochs, learning rate 0.1, batch size 125, decay at epoch 40,
  no weight decay or victim augmentation. Victim IDs are 0 through 5.
- Original `base_dist=cosine`, margin weight 1; the new score determines
  selection independently of the BASIS margin weight. Jacobian addition off.

The eight pinned targets are the first eight IDs in the existing table's files:

- BP: `2540, 8129, 3725, 5705, 7663, 3875, 5169, 6335`.
- GM and SAPA: `843, 1257, 8252, 8730, 8055, 5859, 8954, 821`.

`--keep_pinned_targets` retains exactly these images for every new method.
The clean baseline is measured and recorded for each target. The older table
contains differing evaluation counts; these jobs do not alter its numbers.

## Implemented scores

All scores are maximized over dog-class training candidates. With
`r_i = softmax(z_i) - one_hot(dog)` and
`r_t = softmax(z_t) - one_hot(dog)`:

| CLI `--sel_component` | Per-surrogate score |
|---|---|
| `classifier` | `(r_i dot r_t) * (h_i dot h_t)` |
| `classifier-norm` | `(r_i dot r_t) * cosine(h_i, h_t)` |
| `target-margin` | `z_bird(x_i) - z_dog(x_i)` |
| `similarity-target-margin` | `standardize(cosine(h_i,h_t)) + standardize(z_bird(x_i)-z_dog(x_i))` |
| `confidence` | `-softmax(z_i)[dog]` |

The classifier variants standardize the complete per-surrogate score before
averaging, matching the existing full-gradient-alignment selector's ensemble
weighting. The similarity-plus-margin score standardizes each term per model,
matching BASIS's convention. Standalone margin and confidence average raw
scores over the 20 models. Classifier-only uses the exact requested weight
gradient formula; no additional classifier-bias interaction is included.
The target-margin label comes from the actual target's ground-truth label.

## Results and resume

Results go to
`/home/mmoslem3/scratch/PoisonBase/score_alternatives_8x6_20260921_result/cell_<ID>/`.
Each cell contains its protocol JSON and a normal named run directory with
`results.csv`, `summary.json`, logs, and per-target poison caches.

The runtime stages data, code, existing model checkpoints, and that cell's
partial results into `$SLURM_TMPDIR`. It syncs only that cell back on exit and
on the pre-timeout signal, five minutes before the allocation ends. Resubmit
the same job to resume any incomplete cell. Completion is checked against the
exact 48 target/victim pairs; a partial run is not reported as complete.
An allocation may need resubmission if 48 evaluations exceed the fixed walltime.

These jobs require the existing table model pools (20 surrogates and six clean
victims). Missing checkpoints produce an explicit error before crafting, rather
than training new shared models within an experiment job. Baseline results and
poison caches in `ours_result/` are not used for resume.

## Validation

```bash
python3 sbatch/score_alternatives_8x6_20260921/validate.py
python sbatch/score_alternatives_8x6_20260921/test_scores.py
```

The first command needs only the Python standard library. The second uses the
repository's pinned PyTorch dependencies and checks formulas against explicit
classifier-weight gradients, normalization, ensemble averaging, selection
direction, true-target-label plumbing, cached resume, all 40 CLI configurations,
and rejection of incomplete or duplicate result rows. It downloads no dataset.
