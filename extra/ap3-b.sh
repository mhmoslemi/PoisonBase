#!/usr/bin/env bash
#
# matched architecture: ConvNetBN and VGG13BN
#
# Shard of ap3-matched.sh, sized to fit one allocation: ~8.1 h on an L40S.
#
#   sh appendix/ap3-b.sh
#   DRY_RUN=1 sh appendix/ap3-b.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env MODELS="ConvNetBN VGG13BN" sh appendix/ap3-matched.sh
