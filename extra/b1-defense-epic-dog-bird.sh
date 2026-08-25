#!/usr/bin/env bash
#
# appendix.tex tab:utility-defense -- EPIC on dog-bird.
#
# Four runs (GM and SAPA, Random and DPP), 25 trials each, replaying poisons already on disk. The defense strength is calibrated on clean data and frozen inside the parent script, so there is nothing to pass in.
#
# Estimated ~5 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/b1-defense-epic-dog-bird.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "b1-defense-epic-dog-bird.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

PAIRS=dog-bird sh appendix/ap5-b.sh
echo "=== b1-defense-epic-dog-bird.sh finished ==="
