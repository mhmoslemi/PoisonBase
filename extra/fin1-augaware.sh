#!/usr/bin/env bash
#
# 1 of 5 -- appendix.tex tab:augmentation-aware (12 cells) + the last cell of
#           tab:computational-cost.
#
# All 8 crafts are on disk; this is victim replays only, plus a one-off memory probe.
# The GM/Crop+Flip row is already filled, so those 4 replays skip themselves.
#
# Remaining: SAPA under Crop+Flip (4 replays, ~0.7 h) and both RandAugment rows
# (4 replays at 645 s/trial, ~11 h). ~12 h total, and each STEP is independently
# resumable -- a killed allocation just needs this file rerun.
#
#   sh appendix/fin1-augaware.sh

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "fin1: no CUDA device visible -- get a GPU allocation first"; exit 1; }

echo "########## tab:computational-cost -- surrogate peak memory ##########"
python appendix/surrogate_mem.py --model ConvNetBN || echo "  (memory probe failed; not fatal)"

echo
echo "########## tab:augmentation-aware -- victim replays ##########"
for STEP in cf ra_gm ra_sapa; do
    echo "===== STEP=$STEP ====="
    STEP="$STEP" sh appendix/ap4-augaware.sh || exit 1
done

echo "=== fin1-augaware.sh finished ==="
