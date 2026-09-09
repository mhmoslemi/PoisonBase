#!/bin/sh
# Submit all 30 coefficient-ablation jobs from the PoisonBase checkout.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
JOB_ROOT="$SOURCE_ROOT/sbatch/distance_margin_coef_30_poisonbase"
LOG_ROOT="$SOURCE_ROOT/sbatch/logs"
EXPECTED_JOBS=30
JOB_LIST=""

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
    [ -z "$JOB_LIST" ] || [ ! -e "$JOB_LIST" ] || unlink "$JOB_LIST"
}
trap cleanup EXIT HUP INT TERM

JOB_LIST=$(mktemp)
[ -d "$JOB_ROOT" ] || die "job directory missing: $JOB_ROOT"
[ -f "$JOB_ROOT/_job.sh" ] || die "shared job runtime missing: $JOB_ROOT/_job.sh"
find "$JOB_ROOT" -maxdepth 1 -type f -name 'dmcoefpb_*.sh' -print | sort > "$JOB_LIST"

count=$(wc -l < "$JOB_LIST" | tr -d ' ')
[ "$count" -eq "$EXPECTED_JOBS" ] ||
    die "expected $EXPECTED_JOBS jobs; found $count"

export SOURCE_ROOT ENV_ACTIVATE PERSIST_DATA_ROOT

if [ "${DRY_RUN:-0}" = 1 ]; then
    while IFS= read -r job; do
        printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"
    done < "$JOB_LIST"
    exit 0
fi

mkdir -p "$LOG_ROOT"
while IFS= read -r job; do
    sbatch --account="$ACCOUNT" "$job"
done < "$JOB_LIST"

