#!/usr/bin/env bash
# Submit all 72 fresh S=A -> V jobs.  The generator itself never submits.
set -euo pipefail

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
count=0
for job in "$job_dir"/job_*.sh; do
    sbatch "$job"
    count=$((count + 1))
done
printf 'submitted %d jobs\n' "$count"
