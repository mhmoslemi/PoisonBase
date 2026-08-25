#!/usr/bin/env bash
#
# appendix.tex tab:cross-dataset -- CIFAR-100 attack instances 3 and 4.
#
# 8 runs. Independent of f08 and f10; all three read the same cached surrogates.
#
# Estimated ~6 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f09-cifar100-inst34.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f09-cifar100-inst34.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

STEP=runs INSTANCES="3 4" sh appendix/ap2-cifar100.sh
echo "=== f09-cifar100-inst34.sh finished ==="
