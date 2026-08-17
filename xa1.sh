#!/usr/bin/env bash
#
# Cross-architecture table, FC / ConvNet block: the two off-diagonal DPP cells
# whose attack+victim architecture is ConvNetBN.
#
#     S = ResNet20BN  ->  A = V = ConvNetBN
#     S = VGG13BN     ->  A = V = ConvNetBN
#
# The third row of that block (S = ConvNetBN) is the diagonal cell and is
# already in the table. The Random column needs no run at all -- random base
# selection never looks at a net, so it cannot depend on S.
#
# Cost, at rates measured in this combo's own b0.005 dog-bird log:
#   craft     5 s/target x 5 targets
#   victim   50 s/trial  x 20 trials  (5 targets x victims 0-3)
#   -> 0.28 h per cell, 0.64 h for the file on one L40S.
#   H100: roughly 0.35 h (~1.8x, extrapolated -- nothing here has been
#   timed on an H100, the honest range is 0.29-0.42 h).
#
# Ask for 2 h. Resumable: rerun the same file after a wall-clock kill
# and it restarts at the first missing trial.

set -u

cd /home/mmoslem3/scratch/attack_if

MODEL=ConvNetBN ATTACK=fc SEL_MODELS="ResNet20BN VGG13BN" sh ./xarch.sh || exit 1

echo "=== xa1.sh finished ==="
