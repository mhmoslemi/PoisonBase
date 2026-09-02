#!/usr/bin/env bash
# Submit the paired exact-g_i^T-g_t and A-MR jobs for ten reproducibly sampled
# dog-bird configurations. The sample was drawn without replacement from
# 3 models x 3 attacks x 3 budgets using random seed 20260902.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JOB_DIR="$SCRIPT_DIR/sbatch/exact_amr_dog_bird"

set -- "$JOB_DIR"/job_*.sh
[ -e "$1" ] || {
    echo "ERROR: no job files found under $JOB_DIR" >&2
    exit 1
}
[ "$#" -eq 20 ] || {
    echo "ERROR: expected 20 job files under $JOB_DIR; found $#" >&2
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
