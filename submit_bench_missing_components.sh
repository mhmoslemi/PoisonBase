#!/usr/bin/env bash
# Submit the 12 missing R, -M, and exact g_i^T g_t cells from bench.tex.
# This script submits jobs only; each job contains exactly one configuration.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JOB_DIR="$SCRIPT_DIR/sbatch/bench_missing_components"
LOG_DIR="$SCRIPT_DIR/sbatch/logs"

set -- "$JOB_DIR"/job_*.sh
[ -e "$1" ] || {
    echo "ERROR: no job files found under $JOB_DIR" >&2
    exit 1
}
[ "$#" -eq 12 ] || {
    echo "ERROR: expected 12 job files under $JOB_DIR; found $#" >&2
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

echo "Submitted 12 jobs: 4 on aip-boyuwang and 8 on aip-yiweilu."
