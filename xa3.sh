#!/usr/bin/env bash
#
# Cross-architecture table, FC / ResNet20BN block: the two off-diagonal DPP cells
# whose attack+victim architecture is ResNet20BN.
#
#     S = ConvNetBN   ->  A = V = ResNet20BN
#     S = VGG13BN     ->  A = V = ResNet20BN
#
# The third row of that block (S = ResNet20BN) is the diagonal cell and is
# already in the table. The Random column needs no run at all -- random base
# selection never looks at a net, so it cannot depend on S.
#
# Cost, at rates measured in this combo's own b0.005 dog-bird log:
#   craft     4 s/target x 5 targets
#   victim   86 s/trial  x 20 trials  (5 targets x victims 0-3)
#   -> 0.48 h per cell, 1.03 h for the file on one L40S.
#   H100: roughly 0.57 h (~1.8x, extrapolated -- nothing here has been
#   timed on an H100, the honest range is 0.47-0.69 h).
#   the ResNet20 victim rate has been seen anywhere from 86 to 130 s/trial
#   depending on how busy the node is; at 130 this file is 1.46 h.
#
# Ask for 2 h. Resumable: rerun the same file after a wall-clock kill
# and it restarts at the first missing trial.

set -u

cd /home/mmoslem3/scratch/attack_if

MODEL=ResNet20BN ATTACK=fc SEL_MODELS="ConvNetBN VGG13BN" sh ./xarch.sh || exit 1

echo "=== xa3.sh finished ==="
