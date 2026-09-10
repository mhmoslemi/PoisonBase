#!/bin/sh
# Submit all 28 new jobs: 24 paired transfer cells + 4 BP diagnostics.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
LOG_ROOT="$SOURCE_ROOT/sbatch/logs"
EXPECTED_JOBS=28
JOB_LIST=""

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
cleanup() { [ -z "$JOB_LIST" ] || [ ! -e "$JOB_LIST" ] || unlink "$JOB_LIST"; }
trap cleanup EXIT HUP INT TERM

JOB_LIST=$(mktemp)
find "$SOURCE_ROOT/sbatch/paired_arch_transfer_20260910" -maxdepth 1 \
    -type f -name 'xfer_*.sh' -print > "$JOB_LIST"
find "$SOURCE_ROOT/sbatch/bp_failure_diagnostic_20260910" -maxdepth 1 \
    -type f -name 'bpdiag_*.sh' -print >> "$JOB_LIST"
sort -o "$JOB_LIST" "$JOB_LIST"
count=$(wc -l < "$JOB_LIST" | tr -d ' ')
[ "$count" -eq "$EXPECTED_JOBS" ] || die "expected $EXPECTED_JOBS jobs; found $count"
export SOURCE_ROOT ENV_ACTIVATE PERSIST_DATA_ROOT

if [ "${DRY_RUN:-0}" = 1 ]; then
    while IFS= read -r job; do printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"; done < "$JOB_LIST"
    exit 0
fi
mkdir -p "$LOG_ROOT"
while IFS= read -r job; do sbatch --account="$ACCOUNT" "$job"; done < "$JOB_LIST"
