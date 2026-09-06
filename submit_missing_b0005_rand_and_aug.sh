#!/usr/bin/env bash
# Submit two VGG/BP RandAugment resumes and the ten missing rho=0.0005 RAND cells.

set -eu

ROOT="${ATTACK_IF_ROOT:-/home/mmoslem3/scratch/attack_if}"
JOB_DIR="$ROOT/sbatch/missing_b0005_rand_and_aug"
LOG_DIR="$ROOT/sbatch/logs"

set -- "$JOB_DIR"/job_*.sh
[ -e "$1" ] || {
    echo "ERROR: no jobs found under $JOB_DIR" >&2
    exit 1
}
[ "$#" -eq 12 ] || {
    echo "ERROR: expected 12 jobs; found $#" >&2
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

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo 'Dry run complete: 2 RandAugment resumes and 10 budget-0.0005 RAND jobs.'
else
    echo 'Submitted 2 RandAugment resumes and 10 missing budget-0.0005 RAND jobs.'
fi
