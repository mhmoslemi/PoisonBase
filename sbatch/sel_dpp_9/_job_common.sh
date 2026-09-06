#!/usr/bin/env bash
# Shared launcher for the nine MODEL x ATTACK sel_dpp.sh jobs.

set -Eeuo pipefail

: "${MODEL:?MODEL must be set by the sbatch wrapper}"
: "${ATTACK:?ATTACK must be set by the sbatch wrapper}"

case "$MODEL" in
    ConvNetBN|ResNet20BN|VGG13BN) ;;
    *) echo "ERROR: unsupported MODEL=$MODEL" >&2; exit 1 ;;
esac

case "$ATTACK" in
    fc|gradmatch|sapa) ;;
    *) echo "ERROR: unsupported ATTACK=$ATTACK" >&2; exit 1 ;;
esac

export PROJECT_ROOT=/home/mmoslem3/scratch/attack_if
export DATA_PATH=/home/mmoslem3/scratch/attack_if/data
export PYTHON_ENV=/home/mmoslem3/ENV
export CLASS_PAIR=frog-airplane

source "$PYTHON_ENV/bin/activate"
cd "$PROJECT_ROOT"

# MODEL and ATTACK are the only experiment knobs changed here. All remaining
# values come directly from the current defaults in sel_dpp.sh.
exec bash ./sel_dpp.sh
