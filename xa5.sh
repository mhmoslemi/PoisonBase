#!/usr/bin/env bash
#
# Cross-architecture table, FC / VGG13 block: the two off-diagonal DPP cells
# whose attack+victim architecture is VGG13BN.
#
#     S = ConvNetBN   ->  A = V = VGG13BN
#     S = ResNet20BN  ->  A = V = VGG13BN
#
# The third row of that block (S = VGG13BN) is the diagonal cell and is
# already in the table. The Random column needs no run at all -- random base
# selection never looks at a net, so it cannot depend on S.
#
# Cost, at rates measured in this combo's own b0.005 dog-bird log:
#   craft     8 s/target x 5 targets
#   victim  101 s/trial  x 20 trials  (5 targets x victims 0-3)
#   -> 0.57 h per cell, 1.21 h for the file on one L40S.
#   H100: roughly 0.67 h (~1.8x, extrapolated -- nothing here has been
#   timed on an H100, the honest range is 0.55-0.81 h).
#
# Ask for 2 h. Resumable: rerun the same file after a wall-clock kill
# and it restarts at the first missing trial.

set -u

cd /home/mmoslem3/scratch/attack_if

MODEL=VGG13BN ATTACK=fc SEL_MODELS="ConvNetBN ResNet20BN" sh ./xarch.sh || exit 1

echo "=== xa5.sh finished ==="
