#!/bin/bash
# Shared fixed protocol for the 30 distance/margin coefficient-ablation jobs.
# Each dmcoef_*.sh file supplies exactly one ATTACK, BUDGETS, and coefficient.

set -Eeuo pipefail

: "${ATTACK:?ATTACK is not set by the one-cell job}"
: "${BUDGETS:?BUDGETS is not set by the one-cell job}"
: "${DISTANCE_MARGIN_COEF:?DISTANCE_MARGIN_COEF is not set by the one-cell job}"

case "$ATTACK" in
    gradmatch|sapa) ;;
    *) echo "ERROR: unsupported ATTACK=$ATTACK" >&2; exit 1 ;;
esac
case "$BUDGETS" in
    0.001|0.005|0.002) ;;
    *) echo "ERROR: unsupported BUDGETS=$BUDGETS" >&2; exit 1 ;;
esac
case "$DISTANCE_MARGIN_COEF" in
    0|0.25|0.5|0.75|1) ;;
    *) echo "ERROR: unsupported DISTANCE_MARGIN_COEF=$DISTANCE_MARGIN_COEF" >&2; exit 1 ;;
esac

export SOURCE_ROOT=/home/mmoslem3/scratch/attack_if
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/data
export PYTHON_ENV=/home/mmoslem3/ENV

export JOB_KIND=attack
export MODEL=ConvNetBN
export CLASS_PAIR=dog-bird
export SELECT=ours
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export NUM_TARGETS=6
export NUM_VICTIMS=6
export RECOMPUTE_DELTAS=0
export ORIGINAL_COMMAND="DISTANCE_MARGIN_COEF=$DISTANCE_MARGIN_COEF USE_JACOBIAN_SCORE=0 MODEL=$MODEL ATTACK=$ATTACK CLASS_PAIR=$CLASS_PAIR BUDGETS=$BUDGETS SELECT=$SELECT NUM_TARGETS=$NUM_TARGETS NUM_VICTIMS=$NUM_VICTIMS sh sel_dpp.sh"

source "$SOURCE_ROOT/sbatch/_job_common.sh"

