#!/usr/bin/env bash
#
# broad: the bird->dog reference row
#
# Shard of ap1-broad.sh, sized to fit one allocation: ~3.4 h on an L40S.
#
#   sh appendix/ap1-d.sh
#   DRY_RUN=1 sh appendix/ap1-d.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env PAIRS="dog-bird" sh appendix/ap1-broad.sh
