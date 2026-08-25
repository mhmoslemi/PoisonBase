#!/usr/bin/env bash
#
# The experiments for the remaining cells of latex/appendix.tex. Nothing else.
#
#   sh appendix-cells.sh
#
#   tab:computational-cost   surrogate peak memory                     ~10 min
#   tab:augmentation-aware   SAPA / RandAugment / DPP  (resumes 20/25)  ~1 h
#   tab:cross-dataset        CIFAR-100 GM / DPP, 3 instances left       ~2 h
#
# ~3 h on an L40S. Skips what is done, resumes what was killed, takes no arguments.

set -u
cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "${DRY_RUN:-}" ]; then
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "appendix-cells.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

echo "########## tab:computational-cost -- surrogate peak memory ##########"
python appendix/surrogate_mem.py --model ConvNetBN || echo "  (memory probe failed; not fatal)"

echo
echo "########## tab:augmentation-aware -- SAPA RandAugment DPP ##########"
STEP=ra_sapa SELS=dpp sh appendix/ap4-augaware.sh

echo
echo "########## tab:cross-dataset -- CIFAR-100 GM DPP ##########"
STEP=runs sh appendix/ap2-cifar100.sh

echo
echo "=== appendix-cells.sh finished ==="
