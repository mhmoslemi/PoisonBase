#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder -- the ResNet20BN/GRADMATCH column, rules 9-11.
#
# ResNet20BN is ~1.2 h a rule here (472 s craft plus five 130 s victims per target), so this column is split three ways.
#
# Estimated ~3.6 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/c8-ladder-resnet-gm-9to11.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "c8-ladder-resnet-gm-9to11.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

MODELS=ResNet20BN ATTACKS=gradmatch CRITS="relevance greedy dpp" sh appendix/fin3-ladder.sh
echo "=== c8-ladder-resnet-gm-9to11.sh finished ==="
