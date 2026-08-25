#!/usr/bin/env bash
#
# appendix.tex tab:augmentation-aware -- GM under RandAugment, random selection.
#
# One run, 25 trials. RandAugment costs 645 s a trial against 38 s for standard augmentation, which is why each of these is its own shard.
#
# Estimated ~4.5 h on an L40S. Self-contained: it takes no arguments, skips any
# unit already on disk, and a killed allocation just needs this same file rerun.
#
#   sh appendix/final/a2-aug-randaug-gm-random.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "a2-aug-randaug-gm-random.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

STEP=ra_gm SELS=random sh appendix/ap4-augaware.sh
echo "=== a2-aug-randaug-gm-random.sh finished ==="
