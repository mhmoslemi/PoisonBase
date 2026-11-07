# Reviewer 3: Meta-Grad and Influence Functions

Run `sh submit_meta_influence_lu.sh` from the project. It submits only these two
rows: **38 jobs: 27 aip-bghaddar, 7 aip-boyuwang, 4 aip-yiweilu;
1 CPU, 6 GB RAM, 1 full L40S each**.

The batch contains 14 scoring jobs and 24 cell evaluation jobs. All 24 cells
use 8 pinned bird targets and 6 independent 50-epoch victims per target: 1,152
victim trainings in total. Models are ConvNetBN and ResNet20BN; attacks are the
existing `fc --fc_mode sample` (the paper's BP runner), GM and SAPA; budgets are
0.002 and 0.005 (100/250 poisons out of 50,000 training images). There are no
surrogate-training jobs, parameter sweeps, or additional table rows.

## Method definitions and provenance

These are ports of the authors' scoring procedures for a sample-selection
comparison, using the attack target with adversarial label dog as the outer
example. They are not the original papers' complete training or attack systems.

- Meta-Grad: [official reweighting implementation](https://github.com/uber-research/learning-to-reweight-examples/blob/master/models/reweight_model.py).
  Differentiate target CE through a weighted SGD step whose sample weights start
  at zero, take the negative weight derivative, clip at zero, and normalize over
  all 5,000 eligible dog candidates. Compute this for each of the 20 existing
  surrogate checkpoints and average the normalized weights. The step size is 1;
  its positive magnitude cancels in this normalization. Ties use training index.
  Score batches are combined before clipping/normalization, so minibatch sums do
  not change comparisons between candidates in different batches.
- Influence Functions: [official HVP/LiSSA implementation](https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py)
  and [official poisoning base selector](https://github.com/kohpangwei/influence-release/blob/master/influence/dataset_poisoning.py#L39).
  For each surrogate and target, approximate `(H + 0.1 I)^-1 g_target` using
  10,000 LiSSA updates with uniformly sampled batches of 32 from the **full
  50,000-image training set**, with one trajectory. Rank candidates by the L1
  norm of the input gradient of their influence, then average scores over 20
  surrogates. This is the poisoning selector in the authors' code, not the
  different signed upweighting-influence score. Parameter scope is the full
  trainable network. BN statistics stay fixed at the checkpoint values for
  both methods. Input derivatives are converted to raw pixel coordinates before
  taking L1 norms, to respect the shared raw-pixel perturbation bound.

LiSSA uses `v + (1 - ridge/scale)*estimate - H_batch*estimate/scale`.
The initial scale is 10. Non-finite or explosively growing iterates cause a
restart with twice the scale, preserving the physical ridge of 0.1 and the
sample stream. Every actual scale and iteration count is saved. The maximum
scale is 1,280; instability beyond it fails the task without accepting scores.
This damping/recursion configuration is an explicit experimental choice, not a
claim that the authors prescribe these numerical values for our architectures.

Rankings are shared between budgets and between attacks with identical targets.
The first eight targets in the existing model-specific BP and GM target files
are pinned in `sbatch/rebuttal_meta_influence/plan.json`; SAPA uses the GM IDs.
Crafting uses the same first five 60-epoch surrogate checkpoints and unchanged
250-step, eight-restart GM/SAPA or single-restart BP routines. Victim SGD uses
50 epochs, LR 0.1, batch 125, momentum 0.9, decay at epoch 40, no weight decay,
and no augmentation. Victim seed is the original `(seed, target, victim)` rule.

## Times, dependencies, and resume

- 22 jobs request 2 hours: two Meta-Grad scoring jobs, four ConvNet influence
  scoring jobs (four targets each), and 16 faster cell evaluations.
- Eight ResNet20 influence scoring jobs (two targets each), and four ResNet20
  GM/SAPA 0.5% cells request 3 hours.
- Four ResNet20 GM/SAPA 0.2% cells request 2.5 hours.

Short GPU measurements in the supplied allocation found full LiSSA update
times on real CIFAR-10 data of 0.00411 s (ConvNet) and 0.01757 s (ResNet20),
projecting about 13.7 and 58.6 minutes per target across 20 surrogates, before
checkpoint/selection overhead. Existing logs show about 134--136
minutes for ten-target, six-victim ResNet20 GM/SAPA 0.2% cells and about 138
minutes for eight-target 0.5% cells. These are timing estimates, not completed
results for the new methods.

Evaluation jobs depend on successful completion of their scoring jobs.
Rerunning the submitter skips completed and active tasks, resubmits unfinished
tasks, and reconnects this batch's queued dependents to the new scorer IDs.
Invalid failed dependencies are cancelled by Slurm, not left pending forever.

Outputs go directly to `rebuttal_meta_influence_result/` on scratch. Scoring
checkpoints save each surrogate's progress, LiSSA estimates, exact RNG state,
and candidate offsets. Completed surrogate scores are never recomputed.
The existing attack runner saves base selections, perturbations and completed
target/victim trials; an interrupted single victim restarts its 50 epochs.
Exit 75 means a deliberate time-budget pause, not a completed experiment.

`jobs/<task>/summary.json` holds each completed cell's ASR/CTA and CSV path.
`.done` is written only after checking the complete 8x6 victim grid or all
required score outputs. Frozen code and checksums under the batch directory
keep later edits to root scripts from changing these runs. No existing
experiment script, submitted job, or rebuttal table is modified by this batch.
