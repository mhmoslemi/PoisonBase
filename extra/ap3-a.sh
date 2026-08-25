#!/usr/bin/env bash
#
# matched architecture: ResNet20BN
#
# Shard of ap3-matched.sh, sized to fit one allocation: ~6.2 h on an L40S.
#
#   sh appendix/ap3-a.sh
#   DRY_RUN=1 sh appendix/ap3-a.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env MODELS="ResNet20BN" sh appendix/ap3-matched.sh
