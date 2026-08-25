#!/usr/bin/env bash
#
# broad: cat->dog, deer->horse
#
# Shard of ap1-broad.sh, sized to fit one allocation: ~6.9 h on an L40S.
#
#   sh appendix/ap1-a.sh
#   DRY_RUN=1 sh appendix/ap1-a.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env PAIRS="dog-cat horse-deer" sh appendix/ap1-broad.sh
