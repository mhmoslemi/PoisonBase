#!/usr/bin/env bash
#
# aug-aware: Crop+Flip replays (all 8)
#
# Shard of ap4-augaware.sh, sized to fit one allocation: ~2.1 h on an L40S.
#
# Depends on: needs ap4-a.sh (the crafts)
#
#   sh appendix/ap4-b.sh
#   DRY_RUN=1 sh appendix/ap4-b.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env STEP=cf sh appendix/ap4-augaware.sh
