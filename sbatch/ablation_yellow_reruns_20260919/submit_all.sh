#!/usr/bin/env bash
# Submit all nine yellow-highlighted ablation reruns.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
count=0
for job in "$SCRIPT_DIR"/job_*.sh; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'sbatch %s\n' "$job"
    else
        sbatch "$job"
    fi
    count=$((count + 1))
done

[ "$count" -eq 9 ] || {
    echo "ERROR: expected 9 jobs, found $count" >&2
    exit 1
}
echo "Prepared $count yellow-cell reruns; walltime 3:20 each."
