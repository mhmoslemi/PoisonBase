#!/usr/bin/env bash
#
# tab:augmentation-robustness -- Cutout -- dog-bird FC 1e-2 + frog-airplane GM (both budgets)
#
# Cells filled by this script (ResNet20BN, Random and DPP side by side):
#
#     dog--bird      & FC & $1{\\times}10^{-2}$                            & Random, DPP & <Cutout>
#     frog--airplane & GM & $2{\\times}10^{-3}$ and $2{\\times}10^{-2}$ & Random, DPP & <Cutout>
#
# REDUCED PROTOCOL: 5 targets x 4 victims = 20 trials per cell, one third of the
# 10 x 6 the undefended sweep used. The 5 are the FIRST 5 of the sorted
# random-vs-dpp intersection in target_sets/aug_<combo>_b<budget>.json, and the
# victims are ids 0..3, so the No Aug. column can be recomputed later from the
# existing attack logs by filtering them to those same targets and victim ids --
# it is deliberately not rerun here.
#
# Nothing is crafted. The saved perturbations are read from ours_result/ and the
# augmentation is resampled on top of them every epoch.
#
# Estimated ~4.6 h (6 runs x 20 trials) on an L40S (ResNet20BN, 130 s/victim measured; Crop+Flip and
# Cutout ~1.06x that, RandAugment ~1.4x).
#
#   sh aug10.sh              # as configured
#   DRY_RUN=1 sh aug10.sh    # print the defense.py commands and stop

MODEL="${MODEL:-ResNet20BN}" \
CLASS_PAIR="dog-bird" ATTACK="fc" \
BUDGETS="0.01" SELS="random dpp" PAIR_SELS="random dpp" \
AUGS="cutout" NUM_TARGETS="${NUM_TARGETS:-5}" NUM_VICTIMS="${NUM_VICTIMS:-4}" \
sh "/home/mmoslem3/scratch/attack_if/aug.sh" || exit 1

MODEL="${MODEL:-ResNet20BN}" \
CLASS_PAIR="frog-airplane" ATTACK="gradmatch" \
BUDGETS="0.002 0.02" SELS="random dpp" PAIR_SELS="random dpp" \
AUGS="cutout" NUM_TARGETS="${NUM_TARGETS:-5}" NUM_VICTIMS="${NUM_VICTIMS:-4}" \
sh "/home/mmoslem3/scratch/attack_if/aug.sh" || exit 1

