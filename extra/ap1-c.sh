#!/usr/bin/env bash
#
# broad: ship->frog, truck->cat
#
# Shard of ap1-broad.sh, sized to fit one allocation: ~6.9 h on an L40S.
#
#   sh appendix/ap1-c.sh
#   DRY_RUN=1 sh appendix/ap1-c.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env PAIRS="frog-ship cat-truck" sh appendix/ap1-broad.sh
