#!/bin/sh
# Run on the local Mac. Transfer only the poison caches needed by the remaining
# RandAugment cells in augment-extra.tex, plus any partial RandAugment state,
# from Killarney to Vulcan through a temporary local staging directory.

set -eu

KILLARNEY_HOST="${KILLARNEY_HOST:-mmoslem3@killarney.alliancecan.ca}"
VULCAN_HOST="${VULCAN_HOST:-mmoslem3@vulcan.alliancecan.ca}"
KILLARNEY_ROOT="${KILLARNEY_ROOT:-/home/mmoslem3/scratch/attack_if}"
VULCAN_ROOT="${VULCAN_ROOT:-/home/mmoslem3/scratch/PoisonBase}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JOB_DIR="$SCRIPT_DIR/sbatch/augment_extra"
STAGE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/augment-artifacts.XXXXXX")
# Keep the OpenSSH socket below macOS's 104-byte Unix-socket limit.
if [ -n "${SSH_CONTROL_PATH:-}" ]; then
    SSH_MASTER_OWNER=0
else
    SSH_CONTROL_PATH="/tmp/axa-$$-%C"
    SSH_MASTER_OWNER=1
fi
SSH_OPTIONS="-o ControlMaster=auto -o ControlPersist=12h -o ControlPath=$SSH_CONTROL_PATH"
POISON_RUNS="$STAGE_DIR/poison-runs.txt"
PARTIAL_RUNS="$STAGE_DIR/partial-runs.txt"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [ "${KEEP_STAGE:-0}" = 1 ]; then
        printf 'Local staging retained at %s\n' "$STAGE_DIR"
    else
        find "$STAGE_DIR" -depth -mindepth 1 -exec rm -rf {} + 2>/dev/null || true
        rmdir "$STAGE_DIR" 2>/dev/null || true
    fi
    if [ "$SSH_MASTER_OWNER" = 1 ]; then
        ssh -o "ControlPath=$SSH_CONTROL_PATH" -O exit "$KILLARNEY_HOST" >/dev/null 2>&1 || true
        ssh -o "ControlPath=$SSH_CONTROL_PATH" -O exit "$VULCAN_HOST" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT HUP INT TERM

for command_name in ssh scp find mktemp sed sort; do
    command -v "$command_name" >/dev/null 2>&1 || die "missing command: $command_name"
done
[ -d "$JOB_DIR" ] || die "augmentation sbatch directory missing: $JOB_DIR"

run_ssh() {
    # Intentional splitting: SSH_OPTIONS contains separate OpenSSH arguments.
    # shellcheck disable=SC2086
    ssh $SSH_OPTIONS "$@"
}

run_scp() {
    # Intentional splitting: SSH_OPTIONS contains separate OpenSSH arguments.
    # shellcheck disable=SC2086
    scp $SSH_OPTIONS "$@"
}

remote_dir_exists() {
    host=$1
    path=$2
    run_ssh "$host" "test -d '$path'"
}

download_dir() {
    relative_path=$1
    source_path="$KILLARNEY_ROOT/$relative_path"
    destination_path="$STAGE_DIR/$relative_path"
    destination_parent=$(dirname "$destination_path")

    mkdir -p "$destination_parent"
    printf 'Killarney -> local: %s\n' "$source_path"
    run_scp -rp "$KILLARNEY_HOST:$source_path" "$destination_parent/"
    [ -d "$destination_path" ] || die "download did not create $destination_path"
    find "$destination_path" -type f -name .lock -exec rm -f {} \;
}

upload_dir() {
    relative_path=$1
    local_path="$STAGE_DIR/$relative_path"
    remote_parent="$VULCAN_ROOT/$(dirname "$relative_path")"

    [ -d "$local_path" ] || die "local directory missing before upload: $local_path"
    run_ssh "$VULCAN_HOST" "mkdir -p '$remote_parent'"
    printf 'local -> Vulcan: %s/%s\n' "$VULCAN_ROOT" "$relative_path"
    run_scp -rp "$local_path" "$VULCAN_HOST:$remote_parent/"
}

copy_required_dir() {
    relative_path=$1
    remote_dir_exists "$KILLARNEY_HOST" "$KILLARNEY_ROOT/$relative_path" || \
        die "required Killarney directory absent: $KILLARNEY_ROOT/$relative_path"
    download_dir "$relative_path"
    upload_dir "$relative_path"
}

copy_optional_dir() {
    relative_path=$1
    if remote_dir_exists "$KILLARNEY_HOST" "$KILLARNEY_ROOT/$relative_path"; then
        download_dir "$relative_path"
        upload_dir "$relative_path"
    else
        printf 'Optional Killarney directory absent; skipping: %s/%s\n' \
            "$KILLARNEY_ROOT" "$relative_path"
    fi
}

verify_delta_count() {
    relative_path=$1
    count=$(run_ssh "$VULCAN_HOST" \
        "find '$VULCAN_ROOT/$relative_path' -maxdepth 1 -type f -name 'delta_*.pt' | wc -l" | \
        tr -d '[:space:]')
    case "$count" in
        ''|*[!0-9]*) die "could not count uploaded perturbations in $relative_path" ;;
    esac
    [ "$count" -ge 5 ] || \
        die "Vulcan has only $count perturbations in $relative_path; expected at least 5"
    printf 'Verified on Vulcan: %s perturbations in %s\n' "$count" "$relative_path"
}

vulcan_delta_count() {
    relative_path=$1
    run_ssh "$VULCAN_HOST" \
        "find '$VULCAN_ROOT/$relative_path' -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null | wc -l" | \
        tr -d '[:space:]'
}

extract_poison_runs() {
    for row in 03 04 06 08 10; do
        representative=$(find "$JOB_DIR" -maxdepth 1 -type f \
            -name "*_r${row}_r_r.sh" -print | head -n 1)
        [ -n "$representative" ] || die "no representative RandAugment job for row $row"
        sed -n "s/^export RUN_[A-Z0-9]*='\\(.*\\)'$/\\1/p" "$representative"
    done | sort -u > "$POISON_RUNS"
}

extract_partial_runs() {
    # Four rows have five missing RandAugment cells. In row 08 only Random is
    # still blank; the other four values were accepted from 18/20 or 19/20.
    {
        for row in 03 04 06 10; do
            find "$JOB_DIR" -maxdepth 1 -type f -name "*_r${row}_*_r.sh" -print
        done
        find "$JOB_DIR" -maxdepth 1 -type f -name '*_r08_r_r.sh' -print
    } | while IFS= read -r job; do
        sed -n "s/^export EXPECTED_DEFENSE_RUN='\\(.*\\)'$/\\1/p" "$job"
    done | sort -u > "$PARTIAL_RUNS"
}

printf 'Checking SSH access...\n'
run_ssh "$KILLARNEY_HOST" "test -d '$KILLARNEY_ROOT'" || \
    die "Killarney root unavailable: $KILLARNEY_ROOT"
run_ssh "$VULCAN_HOST" "test -d '$VULCAN_ROOT'" || \
    die "Vulcan root unavailable: $VULCAN_ROOT"

extract_poison_runs
extract_partial_runs

poison_count=$(wc -l < "$POISON_RUNS" | tr -d '[:space:]')
partial_count=$(wc -l < "$PARTIAL_RUNS" | tr -d '[:space:]')
[ "$poison_count" -eq 25 ] || die "expected 25 unique poison runs; found $poison_count"
[ "$partial_count" -eq 21 ] || die "expected 21 unfinished RandAugment cells; found $partial_count"

printf 'Transferring %s unique poison caches...\n' "$poison_count"
while IFS= read -r run_name; do
    relative_path="ours_result/$run_name/poison_cache"
    existing_deltas=$(vulcan_delta_count "$relative_path")
    if [ "$existing_deltas" -ge 5 ]; then
        printf 'Already on Vulcan; skipping: %s (%s perturbations)\n' \
            "$relative_path" "$existing_deltas"
    else
        copy_required_dir "$relative_path"
    fi
    verify_delta_count "$relative_path"
done < "$POISON_RUNS"

printf 'Transferring available partial state for %s RandAugment cells...\n' "$partial_count"
while IFS= read -r run_name; do
    copy_optional_dir "augment_extra_result/$run_name"
done < "$PARTIAL_RUNS"

# These two shared caches are optional, but transferring them avoids retraining
# the clean augmented victims separately in every resumed cell.
copy_optional_dir 'cache/defended_victims/ConvNetBN_none+aug-randaug_50ep_lr0.1_bs125_wd0_seed42'
copy_optional_dir 'cache/defended_victims/ResNet20BN_none+aug-randaug_50ep_lr0.1_bs125_wd0_seed42'

printf '\nTransfer complete: 25 poison caches and all available partial RandAugment state are on Vulcan.\n'
