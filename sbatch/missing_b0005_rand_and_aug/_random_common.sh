#!/usr/bin/env bash
# Shared launcher for the ten missing RAND cells at rho=0.0005.

set -Eeuo pipefail

: "${MODEL:?MODEL must be set by the sbatch wrapper}"
: "${ATTACK:?ATTACK must be set by the sbatch wrapper}"
: "${CLASS_PAIR:?CLASS_PAIR must be set by the sbatch wrapper}"

case "${MODEL}|${ATTACK}|${CLASS_PAIR}" in
    ResNet20BN\|fc\|dog-bird|ResNet20BN\|gradmatch\|dog-bird|ResNet20BN\|sapa\|dog-bird|\
    ResNet20BN\|fc\|frog-airplane|ResNet20BN\|gradmatch\|frog-airplane|ResNet20BN\|sapa\|frog-airplane|\
    VGG13BN\|fc\|dog-bird|VGG13BN\|gradmatch\|dog-bird|VGG13BN\|sapa\|dog-bird|\
    VGG13BN\|fc\|frog-airplane) ;;
    *) echo "ERROR: this is not one of the ten missing RAND cells: ${MODEL}|${ATTACK}|${CLASS_PAIR}" >&2; exit 1 ;;
esac

export PROJECT_ROOT=/home/mmoslem3/scratch/attack_if
export DATA_PATH=/home/mmoslem3/scratch/attack_if/data
export PYTHON_ENV=/home/mmoslem3/ENV
export OUT_DIR=ours_result
export CACHE_DIR=./cache
export SELECT=random
export BUDGETS=0.0005
export CRAFT_STEPS=750
export CRAFT_ALPHA=0.0039216
export FC_RESTARTS=1
export NUM_TARGETS=8
export NUM_VICTIMS=5
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export RECOMPUTE_DELTAS=0
export FORCE=0
unset TARGET_SELECT

source "$PYTHON_ENV/bin/activate"
cd "$PROJECT_ROOT"

# SAPA deliberately shares GradMatch's pinned targets in sel_dpp.sh.
target_attack="$ATTACK"
[ "$target_attack" = sapa ] && target_attack=gradmatch
target_file="target_sets/${MODEL}_${target_attack}_${CLASS_PAIR}.json"
[ -s "$target_file" ] || {
    echo "ERROR: pinned target set is missing: $PROJECT_ROOT/$target_file" >&2
    exit 1
}

exec bash ./sel_dpp.sh
