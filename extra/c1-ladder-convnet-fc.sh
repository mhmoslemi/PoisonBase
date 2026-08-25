#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder -- the ConvNetBN/FC column, all 11 rules.
#
# FC crafting on ConvNetBN is a couple of seconds a target, so the whole column fits one allocation.
#
# Estimated ~4 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/c1-ladder-convnet-fc.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "c1-ladder-convnet-fc.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

MODELS=ConvNetBN ATTACKS=fc sh appendix/fin3-ladder.sh
echo "=== c1-ladder-convnet-fc.sh finished ==="
