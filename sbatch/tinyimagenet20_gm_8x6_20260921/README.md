# Small TinyImageNet appendix experiment

On PoisonBase/Vulcan, submit the entire batch with:

```bash
bash /home/mmoslem3/scratch/PoisonBase/submit_tinyimagenet20_gm_yiweilu.sh
```

For Wang instead, use `submit_tinyimagenet20_gm.sh`. These are alternative
account launchers for the same comparison; running both duplicates the work.
Each account has its own output directory.

This deliberately reduced experiment uses **TinyImageNet, not ImageNet-100
from ImageNet-1K**. Report it with the settings below, including the subsampling.

| Setting | Value |
|---|---|
| Source | `/home/mmoslem3/scratch/PoisonBase/data/tinyimagenet.pt` |
| Classes | Fixed 20 of 200, sampled from sorted WNIDs with seed 0 |
| Training data | Fixed 200 images per class: 4,000 total (4% of the full archive) |
| Validation data | All 50 validation images from each selected class: 1,000 total |
| Input/model | Native 64×64, ResNet18 from scratch, 3×3 stride-1 stem, no initial maxpool |
| Training | 30 epochs, batch 128, SGD momentum 0.9, weight decay 1e-4 |
| Learning rate | 0.1, multiplied by 0.1 after epochs 15 and 25 |
| Augmentation | Random crop 64 with padding 4, horizontal flip; evaluation at native 64 |
| Normalization | ImageNet mean/std, matching the repository's TinyImageNet loader |
| Attack | GM, existing signed Adam, 250 steps, 8 restarts, step size 0.0039216, DiffAugment |
| Poison budget | 0.2% of the 4,000-image training subset: 8 poisons |
| Pixel bound | 8/255, injected before augmentation at native resolution |
| Methods | RAND, M-only, BASIS (margin weight 1, cosine similarity) |
| Shared surrogates | 3 independent clean models, seeds 1042–1044, used by all methods |
| Trials | Same 8 targets × same 6 victim seeds per target, per method |

The class pair is a separate seed-0 draw from the selected classes, fixed before
training. Training examples are sampled with a per-class seed equal to the
original class index. Classes, original sample indices, pair, selected-pixel
checksum, and the full protocol are saved in `manifest.json`; class names are
also saved in `classes_seed0.txt`. Targets are drawn with seed 0 from the fixed
target class's correctly classified validation examples under the shared clean
ensemble. Fewer than eight eligible targets causes a clear failure rather than
changing the pair after observing results.

The launcher validates the archive before any submission. A CPU preparation job
memory-maps the uint8 archive and exports only the selected images losslessly.
GPU jobs stage just this small export, never the entire archive. Preparation is
followed by three surrogate jobs, one target-pinning job, then three experiment
jobs—one method per job, 48 trials each. Failed prerequisites cancel their
dependents instead of leaving them in `DependencyNeverSatisfied`.

Every allocation is **03:20:00**. GPU stages request one L40S, 8 CPUs, and 32 GB;
preparation requests 2 CPUs and 8 GB. Jobs activate
`/home/mmoslem3/ENV/bin/activate`. If a method needs longer, it checkpoints and
requeues the same job, preserving optimizer state and training/crafting progress.
Thus the allocation limit is not a promise that all 48 trials finish in one
allocation. The reduced data, native resolution, and 30-epoch training lower
cost; the GM optimizer and restart count remain unchanged.

Results are under `tinyimagenet20_gm_8x6_20260921_result_yiweilu/` or the same
name without `_yiweilu` for Wang. Each method writes per-trial JSON,
`results.csv`, and `summary.json`, including clean validation accuracy and ASR
mean/std over eight per-target success rates. This is a small supplementary
comparison; it does not establish results for full TinyImageNet or ImageNet-1K.

`DRY_RUN=1 bash ...` previews submission. CPU tests are in
`experiments/test_tinyimagenet20_gm.py` and `experiments/test_imagenet100_gm.py`.
