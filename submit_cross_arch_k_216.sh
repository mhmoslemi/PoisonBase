#!/usr/bin/env bash
# Submit all 216 one-cell BASIS cross-architecture/K jobs.

set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
JOB_ROOT="$ROOT/sbatch/cross_arch_k_10x6_20260915"
LOG_ROOT="$ROOT/sbatch/logs"
EXPECTED_JOBS=216
EXPECTED_BOYU=144
EXPECTED_YIWEI=72

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

main() {
    local list count boyu yiwei job
    list="$(mktemp)"
    trap 'unlink "$list" 2>/dev/null || true' EXIT HUP INT TERM

    [ -f "$JOB_ROOT/_job_common.sh" ] || die "missing runtime: $JOB_ROOT/_job_common.sh"
    [ -f "$JOB_ROOT/MANIFEST.tsv" ] || die "missing manifest: $JOB_ROOT/MANIFEST.tsv"
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'job_*.sh' -print | sort > "$list"
    count="$(wc -l < "$list" | tr -d ' ')"
    [ "$count" -eq "$EXPECTED_JOBS" ] || \
        die "expected $EXPECTED_JOBS job scripts, found $count"

    boyu="$(xargs grep -l '^#SBATCH --account=aip-boyuwang$' < "$list" | wc -l | tr -d ' ')"
    yiwei="$(xargs grep -l '^#SBATCH --account=aip-yiweilu$' < "$list" | wc -l | tr -d ' ')"
    [ "$boyu" -eq "$EXPECTED_BOYU" ] || \
        die "expected $EXPECTED_BOYU aip-boyuwang jobs, found $boyu"
    [ "$yiwei" -eq "$EXPECTED_YIWEI" ] || \
        die "expected $EXPECTED_YIWEI aip-yiweilu jobs, found $yiwei"

    while IFS= read -r job; do
        bash -n "$job" || die "syntax error: $job"
    done < "$list"
    bash -n "$JOB_ROOT/_job_common.sh"

    export ROOT ENV_ACTIVATE DATA_ROOT
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'validated %d jobs: %d Boyu, %d Yiwei\n' "$count" "$boyu" "$yiwei"
        while IFS= read -r job; do
            printf 'sbatch %q\n' "$job"
        done < "$list"
        exit 0
    fi

    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available"
    mkdir -p "$LOG_ROOT"
    printf 'submitting %d jobs: %d Boyu, %d Yiwei\n' "$count" "$boyu" "$yiwei"
    while IFS= read -r job; do
        sbatch "$job"
    done < "$list"
}

main "$@"
