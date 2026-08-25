#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder -- the ConvNetBN/GM column, all 11 rules.
#
# GM crafting is ~100 s a target at this budget; the column still fits one allocation.
#
# Estimated ~5.5 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/c2-ladder-convnet-gm.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "c2-ladder-convnet-gm.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

MODELS=ConvNetBN ATTACKS=gradmatch sh appendix/fin3-ladder.sh
echo "=== c2-ladder-convnet-gm.sh finished ==="
