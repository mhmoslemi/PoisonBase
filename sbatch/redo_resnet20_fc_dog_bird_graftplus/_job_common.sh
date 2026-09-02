#!/usr/bin/env bash
# Shared configuration for the two ResNet20BN/FC/dog-bird GRAFT+ redo jobs.

set -Eeuo pipefail

# Load the user-editable FC crafting knobs before constructing the logged command.
source "$SOURCE_ROOT/sbatch/redo_resnet20_fc_dog_bird_graftplus/fc_hyperparameters.sh"

export PERSIST_DATA_ROOT="$SOURCE_ROOT/data"
export JOB_KIND=attack

# GRAFT+ is the standard GRAFT score with beta=1 for the exact backbone term.
export SELECT=ours
export USE_JACOBIAN_SCORE=1
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64

export MODEL=ResNet20BN
export ATTACK=fc
export CLASS_PAIR=dog-bird
export SEL_ALPHA=2.0
export TARGET_SELECT=''
export NUM_TARGETS=8
export NUM_VICTIMS=5

# These are intentional full redos of existing GRAFT+ cells.  --FORCE makes
# final_update.py ignore prior results, reselect bases, and recraft the poisons.
export FORCE=1
export RECOMPUTE_DELTAS=0

export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export ORIGINAL_COMMAND="SOURCE_ROOT=$SOURCE_ROOT PERSIST_DATA_ROOT=$PERSIST_DATA_ROOT MODEL=$MODEL ATTACK=$ATTACK CLASS_PAIR=$CLASS_PAIR BUDGETS=$BUDGETS SELECT=$SELECT USE_JACOBIAN_SCORE=$USE_JACOBIAN_SCORE JACOBIAN_WEIGHT=$JACOBIAN_WEIGHT CRAFT_STEPS=$CRAFT_STEPS CRAFT_ALPHA=$CRAFT_ALPHA FC_RESTARTS=$FC_RESTARTS FORCE=$FORCE NUM_TARGETS=$NUM_TARGETS NUM_VICTIMS=$NUM_VICTIMS sh sel_dpp.sh"

source "$SOURCE_ROOT/sbatch/_job_common.sh"
