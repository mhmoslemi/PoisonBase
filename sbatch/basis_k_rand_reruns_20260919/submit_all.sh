#!/usr/bin/env bash
# Submit the 22 K-pattern reruns and five non-overlapping BASIS<RAND reruns.

set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
JOB_DIR="$ROOT/sbatch/basis_k_rand_reruns_20260919"
jobs=("$JOB_DIR"/job_*.sh)

if (( ${#jobs[@]} != 27 )); then
    printf 'ERROR: expected 27 job files, found %d\n' "${#jobs[@]}" >&2
    exit 1
fi

for job in "${jobs[@]}"; do
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'sbatch %q\n' "$job"
    else
        sbatch "$job"
    fi
done

printf 'Prepared 27 jobs: 22 K-pattern + 1 ensemble<RAND + 4 main<RAND.\n'
