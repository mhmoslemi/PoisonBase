#!/usr/bin/env bash
#
# tab:defense-robustness -- EPIC, dog-bird
#
# Fills these cells (ResNet20BN, GradMatch, Random and DPP side by side):
#
#     dog--bird & GM & $2{\\times}10^{-3}$ & .. & .. & <Random> & <DPP> & .. & .. \\\\
#     dog--bird & GM & $2{\\times}10^{-2}$ & .. & .. & <Random> & <DPP> & .. & .. \\\\
#
# The No Defense columns are NOT run here: those numbers already exist. Only
# --defense epic is run. Nothing is crafted -- the saved perturbations come
# straight out of ours_result/, exactly as in the undefended runs, so the only
# thing that differs is what the victim does during training.
#
# NOTE on the budget. The table draft says 1e-2 for the second GM row. There are
# no DPP poisons on disk for gradmatch at 1e-2 (dog-bird: 2 targets,
# frog-airplane: 0, on every model), so it is run at 2e-2, where all 10 paired
# targets exist. Change BUDGETS if you craft the missing ones.
#
# Estimated ~23 h (4 runs x 10 targets x 6 victims = 240 EPIC victims) on an L40S. See the header of def3.sh for where that comes from.
#
#   sh def1.sh                      # as configured
#   DRY_RUN=1 sh def1.sh            # print the defense.py commands and stop

sh "/home/mmoslem3/scratch/attack_if/preflight_cuda.sh" || exit 1

for pair in dog-bird; do
    MODEL="${MODEL:-ResNet20BN}" \
    ATTACK=gradmatch \
    CLASS_PAIR="$pair" \
    BUDGETS="${BUDGETS:-0.002 0.02}" \
    SELS="${SELS:-random dpp}" \
    DEFENSES="epic" \
    NUM_VICTIMS="${NUM_VICTIMS:-6}" \
    sh "/home/mmoslem3/scratch/attack_if/defense.sh" || exit 1
done
