#!/usr/bin/env bash
# Shared launcher for the 18 budget-0.0002 sel_dpp.sh jobs.

set -Eeuo pipefail

: "${MODEL:?MODEL must be set by the sbatch wrapper}"
: "${ATTACK:?ATTACK must be set by the sbatch wrapper}"
: "${CLASS_PAIR:?CLASS_PAIR must be set by the sbatch wrapper}"
: "${BUDGETS:?BUDGETS must be set by the sbatch wrapper}"

case "$MODEL" in
    ConvNetBN|ResNet20BN|VGG13BN) ;;
    *) echo "ERROR: unsupported MODEL=$MODEL" >&2; exit 1 ;;
esac

case "$ATTACK" in
    fc|gradmatch|sapa) ;;
    *) echo "ERROR: unsupported ATTACK=$ATTACK" >&2; exit 1 ;;
esac

case "$CLASS_PAIR" in
    dog-bird|frog-airplane) ;;
    *) echo "ERROR: unsupported CLASS_PAIR=$CLASS_PAIR" >&2; exit 1 ;;
esac

[ "$BUDGETS" = 0.0002 ] || {
    echo "ERROR: this job set requires BUDGETS=0.0002, got $BUDGETS" >&2
    exit 1
}

export PROJECT_ROOT=/home/mmoslem3/scratch/PoisonBase
export DATA_PATH=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source "$PYTHON_ENV/bin/activate"
cd "$PROJECT_ROOT"

# MODEL, ATTACK, CLASS_PAIR, and BUDGETS define the matrix cell. All remaining
# experiment values come directly from the current defaults in sel_dpp.sh.
exec bash ./sel_dpp.sh
