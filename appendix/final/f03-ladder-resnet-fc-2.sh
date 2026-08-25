#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder -- the ResNet20BN/FC column, rules 9-11.
#
# Completes the ResNet20BN/FC column. Needs nothing from f02; the two shards are
# independent and can run at the same time on different allocations.
#
# Estimated ~2.7 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f03-ladder-resnet-fc-2.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f03-ladder-resnet-fc-2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

MODELS=ResNet20BN ATTACKS=fc CRITS="relevance greedy dpp" sh appendix/fin3-ladder.sh
echo "=== f03-ladder-resnet-fc-2.sh finished ==="
