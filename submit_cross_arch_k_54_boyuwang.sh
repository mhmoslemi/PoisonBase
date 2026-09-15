#!/usr/bin/env bash
# Submit only the 54 PoisonBase jobs moved from Yiwei to Boyu.

set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
JOB_ROOT="$ROOT/sbatch/cross_arch_k_10x6_boyu_resubmit_20260915"
LOG_ROOT="$ROOT/sbatch/logs"
EXPECTED_JOBS=54

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

main() {
    local list count boyu job
    list="$(mktemp)"
    trap 'unlink "$list" 2>/dev/null || true' EXIT HUP INT TERM

    [ -f "$JOB_ROOT/_job_common.sh" ] || die "missing runtime: $JOB_ROOT/_job_common.sh"
    [ -f "$JOB_ROOT/MANIFEST.tsv" ] || die "missing manifest: $JOB_ROOT/MANIFEST.tsv"
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'job_*.sh' -print | sort > "$list"
    count="$(wc -l < "$list" | tr -d ' ')"
    [ "$count" -eq "$EXPECTED_JOBS" ] || \
        die "expected $EXPECTED_JOBS replacement jobs, found $count"
    boyu="$(xargs grep -l '^#SBATCH --account=aip-boyuwang$' < "$list" | wc -l | tr -d ' ')"
    [ "$boyu" -eq "$EXPECTED_JOBS" ] || \
        die "expected every replacement job to use aip-boyuwang; found $boyu/$count"

    while IFS= read -r job; do
        bash -n "$job" || die "syntax error: $job"
    done < "$list"
    bash -n "$JOB_ROOT/_job_common.sh"

    export ROOT ENV_ACTIVATE DATA_ROOT
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'validated %d Boyu replacement jobs\n' "$count"
        while IFS= read -r job; do
            printf 'sbatch --account=aip-boyuwang %q\n' "$job"
        done < "$list"
        exit 0
    fi

    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available"
    mkdir -p "$LOG_ROOT"
    printf 'submitting %d replacement jobs with aip-boyuwang\n' "$count"
    while IFS= read -r job; do
        sbatch --account=aip-boyuwang "$job"
    done < "$list"
}

main "$@"
