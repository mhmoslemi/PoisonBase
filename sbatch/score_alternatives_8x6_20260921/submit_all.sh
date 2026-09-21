#!/usr/bin/env bash
# Prepare only the five NEW methods; no existing baseline jobs are submitted.
set -Eeuo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$SCRIPT_DIR/validate.py"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p /home/mmoslem3/scratch/PoisonBase/sbatch/logs
fi
count=0
for job in "$SCRIPT_DIR"/job_*.sh; do
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'sbatch %q\n' "$job"
    else
        job_id=$(sbatch --parsable "$job")
        printf '%s -> %s\n' "$(basename "$job")" "$job_id"
    fi
    count=$((count + 1))
done
printf 'Prepared %d new-selector jobs on aip-boyuwang; 03:20:00 each; 8 x 6 trials.\n' "$count"
