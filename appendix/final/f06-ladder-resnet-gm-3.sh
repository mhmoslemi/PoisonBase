#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder -- the ResNet20BN/GRADMATCH column, rules 9-11.
#
# Completes tab:selection-ladder. Once f02-f06 have all landed, the four-blank
# average sentence under the table can finally be written; it needs all four
# model-attack columns and cannot be filled from a partial table.
#
# Estimated ~4.2 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f06-ladder-resnet-gm-3.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f06-ladder-resnet-gm-3.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

MODELS=ResNet20BN ATTACKS=gradmatch CRITS="relevance greedy dpp" sh appendix/fin3-ladder.sh
echo "=== f06-ladder-resnet-gm-3.sh finished ==="
