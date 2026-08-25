#!/usr/bin/env bash
#
# app-base.tex tab:selection-ladder -- the ResNet20BN/FC column, rules 5-8.
#
# Bottom-m, First-m, Random and GraNd are already on disk. A ResNet20BN rule is
# ~54 min here (FC crafting is 5 s, so the five victims per target dominate),
# which is why the seven remaining rules are split across f02 and f03.
#
# Estimated ~3.6 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/f02-ladder-resnet-fc-1.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "f02-ladder-resnet-fc-1.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

MODELS=ResNet20BN ATTACKS=fc CRITS="el2n boundary pixel featsim" sh appendix/fin3-ladder.sh
echo "=== f02-ladder-resnet-fc-1.sh finished ==="
