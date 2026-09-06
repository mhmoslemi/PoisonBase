#!/usr/bin/env bash
# Submit the 3 attacks x 3 networks as nine independent three-hour jobs.

set -eu

PROJECT_ROOT=/home/mmoslem3/scratch/attack_if
JOB_DIR="$PROJECT_ROOT/sbatch/sel_dpp_9"
LOG_DIR="$PROJECT_ROOT/sbatch/logs"

set -- "$JOB_DIR"/job_*.sh
[ -e "$1" ] || {
    echo "ERROR: no sel_dpp job files found under $JOB_DIR" >&2
    exit 1
}
[ "$#" -eq 9 ] || {
    echo "ERROR: expected 9 sel_dpp jobs under $JOB_DIR; found $#" >&2
    exit 1
}

mkdir -p "$LOG_DIR"

for job in "$@"; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch $job"
    else
        sbatch "$job"
    fi
done

echo "Prepared 9 sel_dpp jobs (3 attacks x 3 networks), each with a 03:00:00 walltime."
