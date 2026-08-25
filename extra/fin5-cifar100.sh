#!/usr/bin/env bash
#
# 5 of 5 -- appendix.tex tab:cross-dataset CIFAR-100 rows (6 cells) and the
#           CIFAR-100 half of tab:additional-target-indices (5 cells).
#
# This replaces fin5-tinyimagenet.sh. TinyImageNet at 64x64 cost 2327 s per victim
# and ~74 h for the table; CIFAR-100 answers the same question -- does the selector
# still help off CIFAR-10 -- at 32x32 for roughly a sixth of that. The candidate
# pool is identical either way, 500 training images per class, so the pool-exhaustion
# argument the appendix makes carries over unchanged.
#
# 20 attack runs: 5 sampled class-pair instances x {GM, SAPA} x {Random, DPP},
# ResNet18BN, poison budget 2e-3 (100 poisons out of a 500-image class).
#
# ~13 h: 20 surrogates (~2 h, one-off) then 20 runs at ~0.55 h. Every unit skips
# itself if already on disk, so a killed allocation just needs this file rerun.
#
#   sh appendix/fin5-cifar100.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "fin5: no CUDA device visible -- get a GPU allocation first"; exit 1; }

# torchvision fetches CIFAR-100 on first use (~169 MB). Do it once, up front, so a
# GPU allocation is not spent watching a download.
python - <<'PY' || exit 1
from torchvision import datasets
d = datasets.CIFAR100('/home/mmoslem3/scratch/data', train=True, download=True)
print('CIFAR-100 ready: %d train images, %d classes' % (len(d), len(d.classes)))
PY

sh appendix/ap2-cifar100.sh || exit 1

echo "=== fin5-cifar100.sh finished ==="
