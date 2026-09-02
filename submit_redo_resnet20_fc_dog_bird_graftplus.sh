#!/usr/bin/env bash
# Submit both ResNet20BN/FC/dog-bird GRAFT+ redo jobs.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JOB_DIR="$SCRIPT_DIR/sbatch/redo_resnet20_fc_dog_bird_graftplus"

set -- "$JOB_DIR"/job_*.sh
[ -e "$1" ] || {
    echo "ERROR: no redo job files found under $JOB_DIR" >&2
    exit 1
}
[ "$#" -eq 2 ] || {
    echo "ERROR: expected 2 redo job files under $JOB_DIR; found $#" >&2
    exit 1
}

mkdir -p "$SCRIPT_DIR/sbatch/logs"

for job in "$@"; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch $job"
    else
        sbatch "$job"
    fi
done
