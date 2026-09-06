#!/usr/bin/env bash
# Submit 2 class pairs x 3 networks x 3 attacks at budget 0.0005.
# dog-bird uses aip-boyuwang; frog-airplane uses aip-yiweilu.

set -eu

PROJECT_ROOT=/home/mmoslem3/scratch/attack_if
JOB_DIR="$PROJECT_ROOT/sbatch/budget_0005_18"
LOG_DIR="$PROJECT_ROOT/sbatch/logs"

set -- "$JOB_DIR"/job_*.sh
[ -e "$1" ] || {
    echo "ERROR: no budget-0.0005 job files found under $JOB_DIR" >&2
    exit 1
}
[ "$#" -eq 18 ] || {
    echo "ERROR: expected 18 budget-0.0005 jobs under $JOB_DIR; found $#" >&2
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

echo "Prepared 18 budget-0.0005 jobs with CRAFT_STEPS=750: 9 on aip-boyuwang and 9 on aip-yiweilu."
