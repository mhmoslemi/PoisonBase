#!/usr/bin/env bash
# Submit every pending Greedy cell in extra-data.tex. Jobs are listed from the
# shortest estimated L40S runtime to the longest; Tiny ImageNet is deliberately
# last. DRY_RUN=1 prints the ordered sbatch commands without submitting them.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
JOB_ROOT="$SOURCE_ROOT/sbatch/extra_data"
mkdir -p "$SOURCE_ROOT/sbatch/logs"

COUNT=$(find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_*.sh' -print | wc -l | tr -d ' ')
[ "$COUNT" -eq 54 ] || {
    echo "expected 54 one-cell jobs under $JOB_ROOT; found $COUNT" >&2
    exit 1
}

echo "extra-data: $COUNT one-cell jobs, shortest first; Tiny ImageNet last"
find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_*.sh' -print | sort | while IFS= read -r job; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch $job"
    else
        sbatch "$job"
    fi
done
