#!/usr/bin/env bash
# Submit a minimal CPU-only job to verify that the aip-yiweilu account is usable.

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
LOG_ROOT="${LOG_ROOT:-$SOURCE_ROOT/logs-proposiot/account-test}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# The submitted allocation runs only this small diagnostic block.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    printf 'account test succeeded\n'
    printf 'job_id=%s\n' "$SLURM_JOB_ID"
    printf 'account=%s\n' "${SLURM_JOB_ACCOUNT:-unknown}"
    printf 'cluster=%s\n' "${SLURM_CLUSTER_NAME:-unknown}"
    printf 'host=%s\n' "$(hostname -f)"
    printf 'started=%s\n' "$(date -Is)"
    exit 0
fi

mkdir -p "$LOG_ROOT"
job_name="test-aip-yiweilu"

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: sbatch --account=%s --job-name=%s --time=00:01:00 ' \
        "$ACCOUNT" "$job_name"
    printf '%s\n' "--cpus-per-task=1 --mem=128M $SOURCE_ROOT/test_aip-yiweilu_account.sh"
    exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable; run this on Vulcan"
case "$(hostname -s)" in
    vulcan*) ;;
    *) die "run this on Vulcan (current host: $(hostname -s))" ;;
esac
[ -f "$SOURCE_ROOT/test_aip-yiweilu_account.sh" ] || \
    die "script missing under SOURCE_ROOT=$SOURCE_ROOT"

output=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --job-name="$job_name" \
    --time=00:01:00 \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=128M \
    --output="$LOG_ROOT/%x-%j.out" \
    "$SOURCE_ROOT/test_aip-yiweilu_account.sh")
job_id=${output%%;*}
case "$job_id" in *[!0-9]*|'') die "could not parse sbatch output: $output" ;; esac

printf 'submitted account test -> job %s\n' "$job_id"
printf 'watch: squeue -j %s\n' "$job_id"
printf 'result: %s/test-aip-yiweilu-%s.out\n' "$LOG_ROOT" "$job_id"
