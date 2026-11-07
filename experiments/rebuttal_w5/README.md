# Reviewer 2 W5: saved-poison defense evaluation

Only GM, dog-to-bird, ConvNetBN/ResNet20BN, .1/.2/.5 percent, Rand/BASIS.
No surrogate training, base selection, poison crafting, controls, table edits,
or submission scripts are included.

The grid and attack sources come directly from
`sbatch/rebuttal_saved_defenses/plan.json`, using that experiment's strict
saved-poison loader and SHA256 identities. BASIS uses 8 targets x 5 victims
per setting; Rand uses 10 x 6, except the two ConvNet .1/.2 percent rows,
which each train one victim for CTA and report ASR zero as previously agreed.
The raw measured success is always preserved. There are 482 final victim
trainings per defense, 1,446 across all three, using 90 saved poisoned datasets.

## Feature models: existing, unfiltered poisoned-data models

Both filters use penultimate features of the existing Telltale **initial
trajectory model**, trained on all 50,000 poisoned training samples. They do
not use the Telltale AE, detection labels, filtered victims, ASD models, clean
victim caches, or attack-surrogate weights. No preliminary training is needed.
All 90 required checkpoints were present when preparing this implementation.

The trajectory protocol is SGD .1, momentum .9, batch 128, decays at 70/90,
no augmentation or weight decay, 120 epochs extended to 160/200/240 as needed
by the existing plateau rule. The detector must have finished before its
checkpoint is accepted. Its model name, poison hash, and epoch are checked;
its file hash is recorded and must remain unchanged on resume. Models and
filters are fixed per poisoned dataset, shared across the victim seeds.

Using these poisoned-data feature models adapts the reference defenses to our
from-scratch setting. In particular, Deep k-NN's released experiments used
clean pretrained features and transfer learning. This evaluation retains its
filtering rule, but does not reproduce that clean-pretraining threat model.

## Algorithms

### Deep k-NN

Reference: <https://github.com/neeharperi/DeepKNNDefense>, commit
`ae19815ee6bd8efdce1a567b9d5ad04dfd01388c`, `dataLoader.py:74-88`.
Raw penultimate features, Euclidean distance, plurality label, lowest-label
tie break, query point included, exactly as the released sklearn classifier
predicts on its fitted training set. Keep a sample iff its label matches that
prediction. k=5,000 for every budget/selector, the paper's full-CIFAR-10
feature-collision setting; it is not chosen using poison identities or outcomes.
Distances and top-k are computed in GPU chunks, without an NxN resident graph.
TF32 matrix multiplication is disabled. Progress is checkpointed per chunk.

### Activation Clustering

Reference: ART commit `3011c66d1c0010de802f77a1b3e20f6bab2faaa6`,
`art/defences/detector/poison/{activation_defence,clustering_analyzer}.py`.
The MIT-licensed clustering/reduction functions and analyzer are vendored in
`reference/`; no ART package installation is needed. The function bodies are
unchanged and checked against the source AST. Use ART `ActivationDefence`
defaults: per-class PCA to 10 dimensions, KMeans with 2 clusters, and the
`smaller` cluster-analysis rule. No exclusionary reclassification is enabled
(the ART default). Seed is fixed to `7300000 + target + class`; library
versions are recorded with benchmark results. Neither filter receives poison
indices or target examples. Progress is saved after each class.

### Adversarial training

Reference: <https://github.com/MadryLab/cifar10_challenge>, commit
`f15682d9f1e26eb47a2d3b371ef8b6c7abcf6276`, `pgd_attack.py` and `config.json`.
Use its CIFAR PGD parameters: L-inf epsilon 8/255, step 2/255, 10 steps,
uniform random start, CE ascent, projection into the ball and valid pixel box.
The ball is centered on the already-poisoned image. This is defender-side
adversarial training, not regeneration of attack poisons. During inner PGD,
parameter gradients and BatchNorm buffer updates are disabled; the outer
training pass updates the model normally. This is a PyTorch architecture
adapter, not a reproduction of Madry's original WideResNet training schedule.

## Final victims, checkpoints, and results

All three defenses use fresh victims with the existing matched 50-epoch recipe:
SGD .1, momentum .9, batch 125, LR decay at epoch 40, no augmentation or weight
decay. Filtering methods train on retained data; adversarial training uses all
50,000 already-poisoned samples. Seed is `4200000 + target*100 + victim`.
CTA is clean accuracy on all 10,000 test samples. ASR tests the pinned bird's
prediction against dog, exactly as the other reviewer defense experiments.

Inside an allocation, after loading the same environment:

```sh
python experiments/rebuttal_w5/run.py --cell conv_basis_b0005 --defense deepknn
python experiments/rebuttal_w5/run.py --cell conv_basis_b0005 --defense activation
python experiments/rebuttal_w5/run.py --cell conv_basis_b0005 --defense advtrain
```

`--targets` and `--victims` can restrict a run to part of its pinned grid.
The same command resumes atomically saved model/optimizer/CPU+CUDA/Python+NumPy
RNG checkpoints, or skips completed victims. Each epoch is printed, including
the completed epoch when resuming. USR1/TERM/INT or `DEFENSE_DEADLINE` requests
a safe stop at the next checkpoint boundary, exit 75. A hard kill loses at
most the current epoch/filter block. Persistent output: `rebuttal_w5_result`.
Run `python experiments/rebuttal_w5/run.py --aggregate` inside an allocation to
write the 36-cell completeness summary. ASR/CTA are emitted only for full grids.

## Checks and timing

`allocated_check.sh <model>` runs GPU checks and one representative BASIS .5%
target: feature extraction, each full-data filter, one full 50-epoch victim per
filter, and five AT epochs with a 50-epoch extrapolation. Outputs are isolated
in `rebuttal_w5_benchmark/<model>` and do not populate production tables.
It never invokes sbatch or scancel. Checks compare GPU k-NN with sklearn,
verify the vendored ART function bodies and filtering results, PGD constraints
and BatchNorm preservation, exact interrupted/resumed SGD and AT, and agreement
of ordinary SGD with `final_update.train_from_scratch` across LR decay.

Measured on 2026-11-07 in the supplied one-CPU, 6G, L40S allocations:

| Component | ConvNetBN | ResNet20BN |
|---|---:|---:|
| Dataset/context setup | 12.3 s | 12.2 s |
| Extract 50,000 penultimate features | 2.6 s | 0.8 s |
| Deep k-NN filtering | 1.3 s | 0.9 s |
| Activation Clustering filtering | 1.3 s | 0.15 s |
| 50-epoch victim after Deep k-NN | 19 s | 96 s |
| 50-epoch victim after Activation Clustering | 30 s | 57 s |
| 50-epoch PGD-10 adversarial victim, estimated from 5 epochs | 6.0 min | 10.5 min |
| Peak process host memory | 2.94 GiB | 2.87 GiB |
| Peak allocated GPU memory | 1.71 GiB | 1.02 GiB |

These are one BASIS .5% target per architecture, not a sweep. Filtering retained
19,069 ConvNet and 49,950 ResNet samples for Deep k-NN; 29,605 and 29,772 for
Activation Clustering. Therefore filtered-victim runtime varies with retained
size. The ConvNet checks had CTA 55.28% (k-NN) and 67.29% (AC): these aggressive
reference settings remove substantial clean data and require review before
the full sweep. No hyperparameters were selected using those outcomes.

Both reference/resume check runs passed. Full measurement artifacts are in
`rebuttal_w5_benchmark/{ConvNetBN,ResNet20BN}/timings.json`, with actual library
versions. Benchmark results and unfinished five-epoch AT checkpoints remain
separate from production output. No full experiment jobs were submitted.
