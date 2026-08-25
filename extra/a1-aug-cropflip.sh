#!/usr/bin/env bash
#
# appendix.tex tab:augmentation-aware -- the last Crop+Flip cells, plus the surrogate peak-memory cell of tab:computational-cost.
#
# Replays the two matched-craft SAPA runs under standard augmentation at 38 s a trial, then measures surrogate training memory, which profile_selection.py cannot because it reads training time off the cache without ever training.
#
# Estimated ~1 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/a1-aug-cropflip.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "a1-aug-cropflip.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

python appendix/surrogate_mem.py --model ConvNetBN || echo "  (memory probe failed; not fatal)"

STEP=cf sh appendix/ap4-augaware.sh
echo "=== a1-aug-cropflip.sh finished ==="
