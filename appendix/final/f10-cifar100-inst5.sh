#!/usr/bin/env bash
#
# appendix.tex tab:cross-dataset -- CIFAR-100 attack instance 5.
#
# 4 runs. Completes tab:cross-dataset, and with it every table in the appendix.
#
# Estimated ~3 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f10-cifar100-inst5.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f10-cifar100-inst5.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

STEP=runs INSTANCES="5" sh appendix/ap2-cifar100.sh
echo "=== f10-cifar100-inst5.sh finished ==="
