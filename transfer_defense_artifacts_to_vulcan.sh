#!/bin/sh
# Run this script on the local Mac. It downloads the remaining defense.tex
# prerequisite from Killarney into a temporary local directory, then uploads
# it to the paths used by the Vulcan sbatch job.

set -eu

KILLARNEY_HOST="${KILLARNEY_HOST:-mmoslem3@killarney.alliancecan.ca}"
VULCAN_HOST="${VULCAN_HOST:-mmoslem3@vulcan.alliancecan.ca}"
KILLARNEY_ROOT="${KILLARNEY_ROOT:-/home/mmoslem3/scratch/attack_if}"
VULCAN_ROOT="${VULCAN_ROOT:-/home/mmoslem3/scratch/PoisonBase}"

# defense.tex: VGG13 / SAPA / dog-bird / b=0.01 / EPIC / Greedy_J.
REGULAR_ATTACK_RUN="${REGULAR_ATTACK_RUN:-CIFAR10_VGG13BN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_jacw1_worst0.05_ce5_tgt50}"
REGULAR_DEFENSE_RUN="${REGULAR_DEFENSE_RUN:-${REGULAR_ATTACK_RUN}__def-epic-s0.1-f2-d10}"
REGULAR_TARGET_FILE="${REGULAR_TARGET_FILE:-def_VGG13BN_sapa_dog-bird_b0.01_jacw1.json}"
REGULAR_DEFENDED_CACHE="${REGULAR_DEFENDED_CACHE:-VGG13BN_epic-s0.1-f2-d10_50ep_lr0.1_bs125_wd0_seed42}"

STAGE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/defense-artifacts.XXXXXX")
# macOS TMPDIR paths are too long for OpenSSH's 104-byte Unix-socket limit.
# Keep only the control socket under the short, system-wide /tmp path.
if [ -n "${SSH_CONTROL_PATH:-}" ]; then
    SSH_MASTER_OWNER=0
else
    SSH_CONTROL_PATH="/tmp/dfa-$$-%C"
    SSH_MASTER_OWNER=1
fi
SSH_OPTIONS="-o ControlMaster=auto -o ControlPersist=12h -o ControlPath=$SSH_CONTROL_PATH"

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

for command_name in ssh scp find mktemp; do
    command -v "$command_name" >/dev/null 2>&1 || die "missing command: $command_name"
done

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

remote_file_exists() {
    host=$1
    path=$2
    run_ssh "$host" "test -f '$path'"
}

download_dir() {
    source_relative=$1
    destination_relative=$2
    source_path="$KILLARNEY_ROOT/$source_relative"
    destination_path="$STAGE_DIR/$destination_relative"
    destination_parent=$(dirname "$destination_path")

    mkdir -p "$destination_parent"
    printf 'Killarney -> local: %s\n' "$source_path"
    run_scp -rp "$KILLARNEY_HOST:$source_path" "$destination_parent/"
    [ -d "$destination_path" ] || die "download did not create $destination_path"

    # A copied lock from an interrupted job must not block the Vulcan resume.
    find "$destination_path" -type f -name .lock -exec rm -f {} \;
}

download_file() {
    relative_path=$1
    source_path="$KILLARNEY_ROOT/$relative_path"
    destination_path="$STAGE_DIR/$relative_path"

    mkdir -p "$(dirname "$destination_path")"
    printf 'Killarney -> local: %s\n' "$source_path"
    run_scp -p "$KILLARNEY_HOST:$source_path" "$destination_path"
    [ -f "$destination_path" ] || die "download did not create $destination_path"
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

upload_file() {
    relative_path=$1
    local_path="$STAGE_DIR/$relative_path"
    remote_parent="$VULCAN_ROOT/$(dirname "$relative_path")"

    [ -f "$local_path" ] || die "local file missing before upload: $local_path"
    run_ssh "$VULCAN_HOST" "mkdir -p '$remote_parent'"
    printf 'local -> Vulcan: %s/%s\n' "$VULCAN_ROOT" "$relative_path"
    run_scp -p "$local_path" "$VULCAN_HOST:$remote_parent/"
}

copy_optional_dir() {
    relative_path=$1
    if remote_dir_exists "$KILLARNEY_HOST" "$KILLARNEY_ROOT/$relative_path"; then
        download_dir "$relative_path" "$relative_path"
        upload_dir "$relative_path"
    else
        printf 'Optional Killarney directory absent; skipping: %s/%s\n' \
            "$KILLARNEY_ROOT" "$relative_path"
    fi
}

copy_optional_file() {
    relative_path=$1
    if remote_file_exists "$KILLARNEY_HOST" "$KILLARNEY_ROOT/$relative_path"; then
        download_file "$relative_path"
        upload_file "$relative_path"
    else
        printf 'Optional Killarney file absent; Vulcan will regenerate it: %s/%s\n' \
            "$KILLARNEY_ROOT" "$relative_path"
    fi
}

verify_delta_count() {
    relative_path=$1
    minimum=$2
    count=$(run_ssh "$VULCAN_HOST" \
        "find '$VULCAN_ROOT/$relative_path' -maxdepth 1 -type f -name 'delta_*.pt' | wc -l" | \
        tr -d '[:space:]')
    case "$count" in
        ''|*[!0-9]*) die "could not count uploaded perturbations in $relative_path" ;;
    esac
    [ "$count" -ge "$minimum" ] || \
        die "Vulcan has only $count perturbations in $relative_path; expected at least $minimum"
    printf 'Verified on Vulcan: %s perturbation files in %s\n' "$count" "$relative_path"
}

vulcan_delta_count() {
    relative_path=$1
    run_ssh "$VULCAN_HOST" \
        "find '$VULCAN_ROOT/$relative_path' -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null | wc -l" | \
        tr -d '[:space:]'
}

printf 'Checking SSH access...\n'
run_ssh "$KILLARNEY_HOST" "test -d '$KILLARNEY_ROOT'" || \
    die "Killarney root unavailable: $KILLARNEY_ROOT"
run_ssh "$VULCAN_HOST" "test -d '$VULCAN_ROOT'" || \
    die "Vulcan root unavailable: $VULCAN_ROOT"

# Upload the indispensable perturbations first. The original interrupted job
# did not always preserve its partial defense_result directory, so that state
# must not prevent the poison cache from reaching Vulcan.
regular_poison="ours_result/$REGULAR_ATTACK_RUN/poison_cache"
regular_defense="defense_result/$REGULAR_DEFENSE_RUN"
existing_deltas=$(vulcan_delta_count "$regular_poison")
if [ "$existing_deltas" -ge 7 ]; then
    printf 'Already on Vulcan; skipping poison transfer: %s (%s perturbations)\n' \
        "$regular_poison" "$existing_deltas"
else
    download_dir "$regular_poison" "$regular_poison"
    upload_dir "$regular_poison"
fi
verify_delta_count "$regular_poison" 7

# Resume state and the pinned target list are useful when present. If absent,
# the one-cell job safely regenerates the target list and reruns all 35 trials.
copy_optional_dir "$regular_defense"
copy_optional_file "target_sets/$REGULAR_TARGET_FILE"
copy_optional_dir "cache/defended_victims/$REGULAR_DEFENDED_CACHE"

printf '\nTransfer complete. The remaining Vulcan defense job can now be submitted.\n'
