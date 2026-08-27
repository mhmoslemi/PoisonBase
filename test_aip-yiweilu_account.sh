#!/usr/bin/env bash
# Submit a minimal CPU-only job to verify that the aip-yiweilu account is usable.

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
LOG_ROOT="${LOG_ROOT:-$SOURCE_ROOT/logs-proposiot/account-test}"
PARTITION="${PARTITION:-}"

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
    [ -z "$PARTITION" ] || printf '%s ' "--partition=$PARTITION"
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

association_state=unknown
if command -v sacctmgr >/dev/null 2>&1; then
    associations=$(sacctmgr -nP show assoc where user="${USER:?USER is unset}" \
        format=Cluster,Account,User,Partition,QOS 2>/dev/null || true)
    printf 'Slurm associations for %s (cluster|account|user|partition|qos):\n' "$USER"
    if [ -n "$associations" ]; then
        printf '%s\n' "$associations"
        if printf '%s\n' "$associations" | \
                awk -F'|' -v wanted="$ACCOUNT" '$2 == wanted { found=1 } END { exit !found }'; then
            association_state=present
        else
            association_state=missing
        fi
    else
        printf '  (Slurm returned no readable associations)\n'
    fi
fi

if command -v sinfo >/dev/null 2>&1; then
    printf 'Available partitions (* is the default):\n'
    sinfo -h -o '  %P' | sort -u
fi

if [ "$association_state" = missing ]; then
    die "$ACCOUNT is visible in CCDB but absent from your Vulcan Slurm associations. "\
"The new CCRI/AIP membership has not been provisioned on this cluster; changing "\
"SBATCH flags cannot fix it. Ask the PI/Alliance support to confirm that your CCRI "\
"is a member of $ACCOUNT on Vulcan, then retry after synchronization."
fi

sbatch_args=(
    --parsable
    --account="$ACCOUNT"
    --job-name="$job_name"
    --time=00:01:00
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=128M
    --output="$LOG_ROOT/%x-%j.out"
)
[ -z "$PARTITION" ] || sbatch_args+=(--partition="$PARTITION")

if ! output=$(sbatch "${sbatch_args[@]}" \
        "$SOURCE_ROOT/test_aip-yiweilu_account.sh" 2>&1); then
    printf '%s\n' "$output" >&2
    if [ "$association_state" = present ] && [ -z "$PARTITION" ]; then
        die "$ACCOUNT exists in Slurm, but the default partition rejected it. "\
"Retry with PARTITION=<allowed-partition>; use the partition column printed above."
    fi
    die "Slurm rejected account=$ACCOUNT${PARTITION:+ partition=$PARTITION}"
fi
job_id=${output%%;*}
case "$job_id" in *[!0-9]*|'') die "could not parse sbatch output: $output" ;; esac

printf 'submitted account test -> job %s\n' "$job_id"
printf 'watch: squeue -j %s\n' "$job_id"
printf 'result: %s/test-aip-yiweilu-%s.out\n' "$LOG_ROOT" "$job_id"
