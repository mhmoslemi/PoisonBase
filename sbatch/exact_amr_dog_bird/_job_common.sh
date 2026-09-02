#!/usr/bin/env bash
# Shared settings for the 20 exact/A-MR dog-bird pilot jobs.

set -Eeuo pipefail

# These pilot jobs live in PoisonBase and stage CIFAR-10 from its data folder.
# Every one of the 20 job files sets SOURCE_ROOT before sourcing this helper.
export PERSIST_DATA_ROOT="$SOURCE_ROOT/data"

export JOB_KIND=attack
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export ORIGINAL_COMMAND="SOURCE_ROOT=$SOURCE_ROOT PERSIST_DATA_ROOT=$PERSIST_DATA_ROOT USE_JACOBIAN_SCORE=0 JACOBIAN_BATCH_SIZE=$JACOBIAN_BATCH_SIZE CLASS_PAIR=$CLASS_PAIR MODEL=$MODEL ATTACK=$ATTACK BUDGETS=$BUDGETS SELECT=$SELECT sh sel_dpp.sh"

source "$SOURCE_ROOT/sbatch/_job_common.sh"
