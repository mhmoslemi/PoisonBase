# ImageNet-100 appendix experiment

One command submits the entire experiment, including shared preparation:

```bash
bash /home/mmoslem3/scratch/PoisonBase/submit_imagenet100_gm.sh
```

The equivalent batch on `aip-yiweilu` is:

```bash
bash /home/mmoslem3/scratch/PoisonBase/submit_imagenet100_gm_yiweilu.sh
```

Both activate `/home/mmoslem3/ENV/bin/activate` inside the jobs. Each initial
batch has eight dependent jobs: subset preparation, three independent clean
surrogates, target pinning, and one experiment job for each of RAND, M-only,
and BASIS. Each method runs eight targets and six victims per target.
The batches have separate result directories and can run concurrently.

## Data

Existing ImageNet-1K files are required. The preparation job looks for
`train/<WNID>/...` and `val/<WNID>/...` below `data/imagenet`, `data/imagenet1k`,
`data/ILSVRC2012`, or `data` in the PoisonBase project. Validation images must
already be organized into class directories. No dataset download is attempted.
If the data are elsewhere, `IMAGENET_ROOT=/path/to/imagenet bash ...` overrides
discovery. The exact location on the server has not been verified locally.

The submitter checks the selected classes' training and validation folders
**before submitting any jobs**, so missing data cannot strand GPU dependencies.
Dependent jobs also use `--kill-on-invalid-dep=yes` to cancel themselves if a
prerequisite fails instead of waiting indefinitely.

From the sorted list of exactly 1000 training class IDs, Python's
`random.Random(0).sample(..., 100)` chooses the subset; its sorted class list
is saved as `classes_seed0.txt`. A separate seed-0 draw chooses the poison and
target classes from this subset. Training/validation file lists, actual poison
count, class pair, and a protocol fingerprint are saved in `manifest.json`.
Eight targets are chosen with seed 0 from correctly classified target-class
validation images using the three-surrogate ensemble, then shared by all methods.

## Screenshot settings

- Standard torchvision ImageNet ResNet18, `weights=None`, 100 outputs, 224 inputs.
- Every surrogate and victim trains from scratch for 90 epochs; SGD with batch
  size 128, momentum 0.9, weight decay 1e-4, learning rate 0.1, divided by ten
  after epochs 30 and 60.
- Training uses random resized crop to 224 and horizontal flip. Evaluation
  resizes the shorter side to 256, then center-crops to 224. ImageNet mean/std.
- Three independent clean surrogates (seeds 1042, 1043, 1044) are shared by all
  methods for selection and crafting. The original clean surrogate checkpoints
  are retained. No pretrained weights are loaded.
- GM uses the existing signed-Adam implementation, 250 iterations, eight restarts,
  step size 0.0039216, and default crafting DiffAugment. Exact micro-batching uses
  two poisons at a time to accommodate 224-pixel second-order gradients.
- Poison count is `round(0.002 * actual_subset_training_size)`; all bases have
  the adversarial label, which is unchanged. The pixel bound is exactly 8/255.
- RAND is uniform; M-only picks the smallest adversarial-class logit margins;
  BASIS uses the existing standardized cosine-distance plus standardized margin
  score with margin weight 1 and no Jacobian term.
- Eight pinned targets, six independent victims each. Identical victim seeds
  are used across the three methods for paired comparisons.

Poison perturbations are optimized on a 224-grid in evaluation-center-crop
coordinates and transported to the native image resolution with bounded
bilinear interpolation. The native image is perturbed **before** training
augmentation; its original dimensions are preserved. Crafting differentiates
through the exact same injection and evaluation transforms. The maximum native
pixel perturbation is checked and recorded. No JPEG recompression or label
change is introduced. This geometry choice is included in the protocol identity.

## Fixed walltime and continuation

Every allocation requests `03:20:00`. GPU stages request one L40S, eight CPUs,
and 64 GB RAM; subset preparation requests two CPUs and 8 GB RAM. A method's
48 full 90-epoch training runs will need multiple allocations. Training resumes
from model/optimizer, epoch, and sample-offset checkpoints. Crafting resumes
from Adam state, restart, iteration, best perturbation, and RNG state.

Ten minutes before timeout, the job requests a checkpoint, syncs its own state
from node-local storage, and invokes Slurm's `scontrol requeue` on itself.
Dependent jobs remain behind `afterok` until their prerequisites actually
finish. Explicit cancellation is not converted into a requeue. This requires
the cluster to permit user requeue, as described in the
[Slurm scontrol documentation](https://slurm.schedmd.com/scontrol.html).

Wang output: `imagenet100_gm_8x6_20260921_result/` under the project.
Yiwei output: `imagenet100_gm_8x6_20260921_result_yiweilu/`.
Each method saves per-trial JSONs, poisons, `results.csv`, and `summary.json`.
ASR mean/std are computed over the eight per-target success rates; each trial
also reports clean validation accuracy after poisoned training. No clean-victim
baseline is inferred from the surrogate accuracies.

Preview either launcher with `DRY_RUN=1`. CPU checks are in
`experiments/test_imagenet100_gm.py`; they require the repository's PyTorch
dependencies but no ImageNet files or GPU. Full ImageNet training has not been
run locally.
