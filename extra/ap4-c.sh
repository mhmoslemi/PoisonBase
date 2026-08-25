#!/usr/bin/env bash
#
# aug-aware: RandAugment replay, GM
#
# Shard of ap4-augaware.sh, sized to fit one allocation: ~5.5 h on an L40S.
#
# Depends on: needs ap4-a.sh (the crafts)
#
#   sh appendix/ap4-c.sh
#   DRY_RUN=1 sh appendix/ap4-c.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env STEP=ra_gm sh appendix/ap4-augaware.sh
