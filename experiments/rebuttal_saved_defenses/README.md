# Saved-poison ASD/Telltale rebuttal batch

Submit from the repository root:

```sh
sh submit_saved_defenses_vulcan.sh
```

There are 36 bundles, each requesting 6 hours, one CPU and one full L40S,
account `aip-bghaddar`: 8 ConvNet ASD bundles, 24 ResNet20 ASD bundles,
2 ConvNet Telltale bundles and 2 ResNet20 Telltale bundles. ASD requests 6G
host memory; Telltale requests 16G. The six-hour duration is a resource limit,
not a runtime guarantee. Retrying continues from persistent checkpoints.

Only batches 7 (ASD) and 32 (Telltale) use the four missing ConvNet Random 0.5
targets 8954, 821, 1547 and 4427. Until all four base/delta pairs are present,
these batches request `afterok:2168638`. The other 34 have no dependency.
If a replacement perturbation job is needed, set `PERTURB_JOB_ID` when invoking
the same submitter. A successful dependency is required; failed upstream jobs
are never treated as usable poisons.

## Scope and data

Only GM on dog-to-bird, ConvNetBN/ResNet20BN, budgets .1/.2/.5 percent and
Random/BASIS. No controls, M/R selectors, surrogate training, attack crafting,
or modifications to the rebuttal or historical results are included.

BASIS uses the first 8 pinned targets and victims 0--4. Nonzero Random rows
use all 10 targets and victims 0--5. The two zero-ASR Random rows (ConvNet
.1/.2) each use one target and one defended victim for CTA. Their reported
ASR is fixed to zero as requested; the raw measured target prediction is
preserved separately. Total per defense: 482 victim trainings. Telltale
detection runs once for each of the 90 used poisoned datasets, shared by
the corresponding victims.

The exact source runs and assignments are frozen in
`sbatch/rebuttal_saved_defenses/plan.json`. Cache lookup is restricted to the
matching run directories in `PoisonBase/ours_result` and
`/scratch/mmoslem3/attack_if/ours_result`. Both per-target and legacy cache
formats are supported. For delta-only ConvNet Random caches, the historical
CPU-generator seed `42*100003 + target_id` reproduces the original base indices.
No new random draw is used. Tensor shapes, poison class, uniqueness and epsilon
are checked. A SHA256 of each base/delta pair locks resumed runs to the same poisons.

Victim seed is the historical `42*100000 + target_id*100 + victim_id`.
CIFAR-10 labels, normalization and architecture definitions match `final_update`.
CTA uses all 10,000 test images. The convergence validation set has 100 test
images per class, excludes the pinned targets and is drawn once with seed 71307.

## Algorithms and explicit adaptations

ASD source: https://github.com/KuofengGao/ASD,
commit `8bcb10b345fa39f239e6c211e8190ec27d88efbd`, especially `ASD.py`,
`utils/trainer/semi.py`, `model/loss.py`, `config/baseline_asd.yaml`.
The implementation retains 60/90/120 stage boundaries, SCE alpha .1/beta 1,
100 clean trusted seeds (10/class, excluded from the actual poison indices),
ten additional low-loss candidates/class every five epochs, lowest-half
class-agnostic split, and lowest-half loss reduction after one virtual Adam
.015 epoch. Main training is Adam .002 with 120 epochs x 1024 MixMatch
iterations, batches of 64 labeled and two views of 64 unlabeled examples,
temperature .5, beta alpha .75, lambda-u 15 with a 120-epoch ramp, BatchNorm
interleaving, reflected crop padding 4 and horizontal flip. The necessary
architecture adaptation replaces the released ResNet18 with the table's models:
the virtual model updates ConvNet's final two blocks/head or ResNet20's
layer2/layer3/head. Augmentation runs on GPU tensors instead of PIL workers.

Telltale sources:
https://github.com/MPaloze/Telltale,
commit `66c55fed231385255e7ed81ca61483583508cbb5`, and
https://www.ndss-symposium.org/wp-content/uploads/2025-1589-paper.pdf,
sections IV-B, V-A3, VII-A. The released LSTM structure, ten full-batch
Adam .0005 updates on 6,000 mixed samples, and model selection on 2,000 mixed
samples are retained. Detector samples are shuffled without poison labels.
FFT uses both amplitude and phase, followed directly by DBSCAN eps 5,
min_samples 4, as stated in the paper. The public visualization demo instead
uses magnitude only, t-SNE, min_samples 5 and oracle clean/poison group
normalization; these differences are not silently carried into this batch.
Normalization is fitted once over all mixed latent vectors. The largest
non-noise cluster is retained. No poison identity enters the detector.
The memory-bounded DBSCAN implementation implements the same Euclidean
core/border/noise rules without storing an entire pairwise neighbor graph.

The paper does not supply a numerical plateau rule or a classifier schedule for
our two architectures. The declared adaptation is SGD .1/momentum .9,
batch 128, no augmentation/WD, decays at 70/90, initially 120 epochs.
A plateau is five validation epochs with relative range <=1%, after epoch 60,
with at least 20 following epochs. If necessary, checkpointed trajectory
training extends to 160/200/240 epochs. Absence of a plateau at that point
is an explicit incomplete defense, never a fabricated ASR result.
Retraining on retained examples uses the original 50-epoch victim recipe:
SGD .1/momentum .9, batch 125, decay at 40, no augmentation or weight decay.

## Resume and results

Run the same submit command after a timeout or failure. It skips completed
bundles and bundles already queued/running, then submits unfinished bundles.
For a single bundle: `sh submit_saved_defenses_vulcan.sh 7`.
No jobs are canceled or automatically submitted by the runner itself.

Checkpoints live in `rebuttal_saved_defenses_result`, not node-local temporary
storage: ASD and SGD save model, optimizer, next epoch, and Python/NumPy/CPU/CUDA
random states every epoch. The LSTM saves every update; clustering saves core
checks and expansion progress. Slurm sends USR1 three minutes before expiry;
the runner saves at its next boundary and exits 75. A hard kill can lose the
current epoch/clustering block but leaves the previous atomic checkpoint usable.
Finished victims and completed detection masks are skipped. A per-bundle lock
prevents simultaneous retries; persistent poison hashes prevent incompatible
resume. Logs print resumed epoch counts, every completed epoch, and final results.

Submission history: `sbatch/rebuttal_saved_defenses/submissions.tsv`.
Logs: `sbatch/rebuttal_saved_defenses/logs`.
Frozen source: `sbatch/rebuttal_saved_defenses/code` (SHA256 checked at job start).
Aggregate on an allocated node with
`python sbatch/rebuttal_saved_defenses/code/run.py --aggregate`; only complete
cells receive final ASR/CTA values in `rebuttal_saved_defenses_result/summary.json`.

The submitter was checked with mocked Slurm commands for all 36 resource
requests, both dependencies, active-job skipping and retry of unfinished work.
Each allocated job first runs tiny synthetic SGD/ASD checkpoint equivalence
checks and an interrupted/resumed DBSCAN comparison against sklearn, entirely
in node-local temporary storage. GPU execution has not been performed for this
new batch before delivery; job durations remain estimates.
