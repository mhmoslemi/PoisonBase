#!/bin/sh
# Submit the 24 one-cell paired architecture-transfer jobs.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
JOB_ROOT="$SOURCE_ROOT/sbatch/paired_arch_transfer_20260910"
LOG_ROOT="$SOURCE_ROOT/sbatch/logs"
EXPECTED_JOBS=24
JOB_LIST=""

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
cleanup() { [ -z "$JOB_LIST" ] || [ ! -e "$JOB_LIST" ] || unlink "$JOB_LIST"; }
trap cleanup EXIT HUP INT TERM

JOB_LIST=$(mktemp)
[ -f "$JOB_ROOT/_job_common.sh" ] || die "missing runtime: $JOB_ROOT/_job_common.sh"
find "$JOB_ROOT" -maxdepth 1 -type f -name 'xfer_*.sh' -print | sort > "$JOB_LIST"
count=$(wc -l < "$JOB_LIST" | tr -d ' ')
[ "$count" -eq "$EXPECTED_JOBS" ] || die "expected $EXPECTED_JOBS jobs; found $count"
export SOURCE_ROOT ENV_ACTIVATE PERSIST_DATA_ROOT

if [ "${DRY_RUN:-0}" = 1 ]; then
    while IFS= read -r job; do printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"; done < "$JOB_LIST"
    exit 0
fi
mkdir -p "$LOG_ROOT"
while IFS= read -r job; do sbatch --account="$ACCOUNT" "$job"; done < "$JOB_LIST"
