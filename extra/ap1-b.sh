#!/usr/bin/env bash
#
# broad: automobile->truck, bird->airplane
#
# Shard of ap1-broad.sh, sized to fit one allocation: ~6.9 h on an L40S.
#
#   sh appendix/ap1-b.sh
#   DRY_RUN=1 sh appendix/ap1-b.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env PAIRS="truck-automobile airplane-bird" sh appendix/ap1-broad.sh
