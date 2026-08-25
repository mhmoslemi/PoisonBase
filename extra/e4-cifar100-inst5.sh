#!/usr/bin/env bash
#
# appendix.tex tab:cross-dataset -- CIFAR-100 attack instance(s) 5.
#
# 4 runs: GM and SAPA, Random and DPP, one pinned target and five victims each. Needs e1 to have cached the surrogates.
#
# Estimated ~3 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/e4-cifar100-inst5.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "e4-cifar100-inst5.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

STEP=runs INSTANCES="5" sh appendix/ap2-cifar100.sh
echo "=== e4-cifar100-inst5.sh finished ==="
