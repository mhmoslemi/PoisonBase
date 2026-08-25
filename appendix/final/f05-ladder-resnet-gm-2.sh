#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder -- the ResNet20BN/GRADMATCH column, rules 5-8.
#
# See f04. Independent of f04 and f06.
#
# Estimated ~5.6 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f05-ladder-resnet-gm-2.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f05-ladder-resnet-gm-2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

MODELS=ResNet20BN ATTACKS=gradmatch CRITS="el2n boundary pixel featsim" sh appendix/fin3-ladder.sh
echo "=== f05-ladder-resnet-gm-2.sh finished ==="
