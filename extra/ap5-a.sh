#!/usr/bin/env bash
#
# defenses: clean-utility calibration sweep
#
# Shard of ap5-utilmatch.sh, sized to fit one allocation: ~2.2 h on an L40S.
#
#   sh appendix/ap5-a.sh
#   DRY_RUN=1 sh appendix/ap5-a.sh

set -u
cd /home/mmoslem3/scratch/attack_if
exec env STAGE=1 sh appendix/ap5-utilmatch.sh
