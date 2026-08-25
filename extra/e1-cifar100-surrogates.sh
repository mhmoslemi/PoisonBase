#!/usr/bin/env bash
#
# appendix.tex tab:cross-dataset -- the 20 CIFAR-100 surrogates, and the backbone-choice probe.
#
# One-off. Every attack run in e2-e4 reads these from the cache, so this has to land first. Also fetches CIFAR-100 if it is not already on disk.
#
# Estimated ~3 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/e1-cifar100-surrogates.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "e1-cifar100-surrogates.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

python - <<'PY' || exit 1
from torchvision import datasets
d = datasets.CIFAR100('/home/mmoslem3/scratch/data', train=True, download=True)
print('CIFAR-100 ready: %d train images, %d classes' % (len(d), len(d.classes)))
PY

STEP=probe      sh appendix/ap2-cifar100.sh || exit 1
STEP=surrogates sh appendix/ap2-cifar100.sh
echo "=== e1-cifar100-surrogates.sh finished ==="
