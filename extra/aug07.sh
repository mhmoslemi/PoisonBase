#!/usr/bin/env bash
#
# tab:augmentation-robustness -- RandAugment + Crop+Flip -- frog-airplane / FC / $2e-3$
#
# Cells filled by this script (ResNet20BN, Random and DPP side by side):
#
#     frog-airplane & FC & $2e-3$ & Random & <Crop+Flip> <RandAugment>
#     frog-airplane & FC & $2e-3$ & DPP    & <Crop+Flip> <RandAugment>
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
# Estimated ~3.6 h (4 runs x 20 trials) on an L40S (ResNet20BN, 130 s/victim measured; Crop+Flip and
# Cutout ~1.06x that, RandAugment ~1.4x).
#
#   sh aug07.sh              # as configured
#   DRY_RUN=1 sh aug07.sh    # print the defense.py commands and stop

MODEL="${MODEL:-ResNet20BN}" \
CLASS_PAIR="frog-airplane" ATTACK="fc" \
BUDGETS="0.002" SELS="random dpp" PAIR_SELS="random dpp" \
AUGS="randaug standard" NUM_TARGETS="${NUM_TARGETS:-5}" NUM_VICTIMS="${NUM_VICTIMS:-4}" \
sh "/home/mmoslem3/scratch/attack_if/aug.sh" || exit 1

