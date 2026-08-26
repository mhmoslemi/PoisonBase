#!/usr/bin/env bash
# Resume the eight interrupted Jacobian-score FC runs.
# Safe to rerun: sel_dpp.sh does not pass --no_resume or --recompute_deltas.

set -u

cd /home/mmoslem3/scratch/attack_if || exit 1

# DPP_J, b=0.005: VGG13BN and ResNet20BN, both class pairs.
MODEL="VGG13BN ResNet20BN" \
ATTACK=fc \
CLASS_PAIR="dog-bird frog-airplane" \
BUDGETS=0.005 \
SELECT=dpp \
SEL_ALPHA=2.0 \
TARGET_SELECT="" \
USE_JACOBIAN_SCORE=1 \
JACOBIAN_WEIGHT=1.0 \
JACOBIAN_BATCH_SIZE=64 \
sh sel_dpp.sh

# Greedy, b=0.01: VGG13BN and ResNet20BN, dog-bird.
MODEL="VGG13BN ResNet20BN" \
ATTACK=fc \
CLASS_PAIR=dog-bird \
BUDGETS=0.01 \
SELECT=ours \
TARGET_SELECT="" \
USE_JACOBIAN_SCORE=1 \
JACOBIAN_WEIGHT=1.0 \
JACOBIAN_BATCH_SIZE=64 \
sh sel_dpp.sh

# Greedy and DPP_J, b=0.01: ConvNetBN, frog-airplane.
MODEL=ConvNetBN \
ATTACK=fc \
CLASS_PAIR=frog-airplane \
BUDGETS=0.01 \
SELECT="ours dpp" \
SEL_ALPHA=2.0 \
TARGET_SELECT="" \
USE_JACOBIAN_SCORE=1 \
JACOBIAN_WEIGHT=1.0 \
JACOBIAN_BATCH_SIZE=64 \
sh sel_dpp.sh

