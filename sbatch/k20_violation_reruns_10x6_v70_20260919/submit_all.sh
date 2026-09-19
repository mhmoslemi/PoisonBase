#!/usr/bin/env bash
# Validate, then submit the 18 K=20 violation reruns.

set -Eeuo pipefail
shopt -s nullglob

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
JOB_DIR="$ROOT/sbatch/k20_violation_reruns_10x6_v70_20260919"
jobs=("$JOB_DIR"/job_*.sh)

"$JOB_DIR/validate.sh"
mkdir -p "$ROOT/sbatch/logs"

for job in "${jobs[@]}"; do
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'sbatch %q\n' "$job"
    else
        job_id=$(sbatch --parsable "$job")
        printf 'submitted %s -> %s\n' "$(basename "$job")" "$job_id"
    fi
done

printf 'Prepared 18 K=20 jobs (10 targets x 6 victims each).\n'
