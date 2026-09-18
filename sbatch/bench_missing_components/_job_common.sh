#!/usr/bin/env bash
# Shared protocol for the 12 missing R, -M, and exact g_i^T g_t cells in
# bench.tex. These settings reproduce the completed 8-target x 5-victim
# component-ablation runs rather than relying on the current sweep defaults.

set -Eeuo pipefail

: "${MODEL:?MODEL must be set by the job file}"
: "${ATTACK:?ATTACK must be set by the job file}"
: "${BUDGETS:?BUDGETS must be set by the job file}"
: "${SELECT:?SELECT must be set by the job file}"

case "$SELECT" in
    r|minus-m|exact) ;;
    *) echo "ERROR: expected SELECT=r, minus-m, or exact; got $SELECT" >&2; exit 1 ;;
esac

export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
export PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-$PYTHON_ENV/bin/activate}"

export JOB_KIND=attack
export CLASS_PAIR=dog-bird
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export DISTANCE_MARGIN_COEF=''
export LAMBDA_MARGIN=1.0
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export CRAFT_STEPS=250
export CRAFT_ALPHA=0.0039216
export NUM_TARGETS=8
export NUM_VICTIMS=5
export RECOMPUTE_DELTAS=0
export FORCE=0

export ORIGINAL_COMMAND="source $ENV_ACTIVATE; SOURCE_ROOT=$SOURCE_ROOT PERSIST_DATA_ROOT=$PERSIST_DATA_ROOT MODEL=$MODEL ATTACK=$ATTACK CLASS_PAIR=$CLASS_PAIR BUDGETS=$BUDGETS SELECT=$SELECT USE_JACOBIAN_SCORE=0 LAMBDA_MARGIN=$LAMBDA_MARGIN CRAFT_STEPS=$CRAFT_STEPS CRAFT_ALPHA=$CRAFT_ALPHA NUM_TARGETS=$NUM_TARGETS NUM_VICTIMS=$NUM_VICTIMS sh sel_dpp.sh"

source "$SOURCE_ROOT/sbatch/_job_common.sh"
