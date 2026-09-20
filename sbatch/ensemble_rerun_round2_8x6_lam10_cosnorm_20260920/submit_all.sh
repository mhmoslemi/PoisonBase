#!/usr/bin/env bash
# Validate, then submit all 34 requested reruns.

set -Eeuo pipefail
shopt -s nullglob

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
JOB_DIR="$ROOT/sbatch/ensemble_rerun_round2_8x6_lam10_cosnorm_20260920"
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

printf 'Submitted 34 fresh ensemble reruns (8 targets x 6 victims each).\n'
