#!/usr/bin/env bash
# Resume the four interrupted ResNet20BN Jacobian-score gradmatch/SAPA runs.
# Safe to rerun: sel_dpp.sh does not pass --no_resume or --recompute_deltas.

set -u

cd /home/mmoslem3/scratch/attack_if || exit 1

MODEL=ResNet20BN \
ATTACK="gradmatch sapa" \
CLASS_PAIR=frog-airplane \
BUDGETS=0.01 \
SELECT="ours dpp" \
SEL_ALPHA=2.0 \
SHARP_MODE=worst \
SHARP_SIGMA=0.05 \
TARGET_SELECT="" \
USE_JACOBIAN_SCORE=1 \
JACOBIAN_WEIGHT=1.0 \
JACOBIAN_BATCH_SIZE=64 \
sh sel_dpp.sh

