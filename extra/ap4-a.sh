#!/usr/bin/env bash
#
# aug-aware: the 8 crafts
#
# Shard of ap4-augaware.sh, sized to fit one allocation: ~5.5 h on an L40S.
#
#   sh appendix/ap4-a.sh
#   DRY_RUN=1 sh appendix/ap4-a.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env STEP=craft sh appendix/ap4-augaware.sh
