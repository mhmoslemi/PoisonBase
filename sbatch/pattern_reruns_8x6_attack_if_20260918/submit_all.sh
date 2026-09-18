#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/attack_if}"
JOB_DIR="$ROOT/sbatch/pattern_reruns_8x6_attack_if_20260918"
jobs=("$JOB_DIR"/job_*.sh)

if (( ${#jobs[@]} != 72 )); then
    printf 'ERROR: expected 72 job files, found %d\n' "${#jobs[@]}" >&2
    exit 1
fi

for job in "${jobs[@]}"; do
    sbatch "$job"
done
