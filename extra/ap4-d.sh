#!/usr/bin/env bash
#
# aug-aware: RandAugment replay, SAPA
#
# Shard of ap4-augaware.sh, sized to fit one allocation: ~5.5 h on an L40S.
#
# Depends on: needs ap4-a.sh (the crafts)
#
#   sh appendix/ap4-d.sh
#   DRY_RUN=1 sh appendix/ap4-d.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env STEP=ra_sapa sh appendix/ap4-augaware.sh
