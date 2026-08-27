#!/bin/sh
# One-shot repair for the current Vulcan submission:
#   1. one ordinary SSH connection to Killarney downloads the missing defense
#      poison cache and any optional resume/cache state as one tar archive;
#   2. one ordinary SSH connection to Vulcan uploads that archive together with
#      the corrected jobs, then optionally submits them.
#
# The 25 augmentation poison caches are not downloaded again: the previous
# augmentation transfer completed and verified them on Vulcan, and the remote
# submitter verifies them again before submitting anything.

set -eu

KILLARNEY_HOST="${KILLARNEY_HOST:-mmoslem3@killarney.alliancecan.ca}"
VULCAN_HOST="${VULCAN_HOST:-mmoslem3@vulcan.alliancecan.ca}"
KILLARNEY_ROOT="${KILLARNEY_ROOT:-/home/mmoslem3/scratch/attack_if}"
VULCAN_ROOT="${VULCAN_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
SUBMIT_AFTER_TRANSFER="${SUBMIT_AFTER_TRANSFER:-0}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
STAGE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/remaining-artifacts.XXXXXX")
MANIFEST="$STAGE_DIR/killarney-paths.txt"
KILLARNEY_ARCHIVE="$STAGE_DIR/killarney-artifacts.tar"

# Explicitly disable any ControlMaster settings inherited from ~/.ssh/config.
# Each host is contacted exactly once by this script.
SSH_OPTIONS="-o ControlMaster=no -o ControlPersist=no -o ControlPath=none"

ATTACK_RUN='CIFAR10_VGG13BN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_jacw1_worst0.05_ce5_tgt50'
DEFENSE_RUN="${ATTACK_RUN}__def-epic-s0.1-f2-d10"
TARGET_FILE='def_VGG13BN_sapa_dog-bird_b0.01_jacw1.json'
DEFENDED_CACHE='VGG13BN_epic-s0.1-f2-d10_50ep_lr0.1_bs125_wd0_seed42'

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
}
trap cleanup EXIT HUP INT TERM

for command_name in ssh tar find mktemp cp; do
    command -v "$command_name" >/dev/null 2>&1 || die "missing command: $command_name"
done
case "$SUBMIT_AFTER_TRANSFER" in
    0|1) ;;
    *) die 'SUBMIT_AFTER_TRANSFER must be 0 or 1' ;;
esac

for required in \
    "$SCRIPT_DIR/sbatch/vulcan_remaining_20260827" \
    "$SCRIPT_DIR/submit_vulcan-remaining.sh" \
    "$SCRIPT_DIR/submit_vulcan-remaining-defense.sh" \
    "$SCRIPT_DIR/submit_vulcan-remaining-augmentation.sh"; do
    [ -e "$required" ] || die "required local file is missing: $required"
done

# The first path is required. The other three are optional and are included
# only if the interrupted Killarney job actually preserved them.
{
    printf 'ours_result/%s/poison_cache\n' "$ATTACK_RUN"
    printf 'defense_result/%s\n' "$DEFENSE_RUN"
    printf 'target_sets/%s\n' "$TARGET_FILE"
    printf 'cache/defended_victims/%s\n' "$DEFENDED_CACHE"
} > "$MANIFEST"

printf 'Authenticate to Killarney once...\n'
# stdin supplies the path list; stdout is one tar archive. Missing optional
# paths are reported on stderr and omitted without aborting the required cache.
# Intentional splitting: SSH_OPTIONS contains separate OpenSSH arguments.
# shellcheck disable=SC2086
ssh $SSH_OPTIONS "$KILLARNEY_HOST" \
    "set -eu
     cd '$KILLARNEY_ROOT'
     while IFS= read -r artifact_path; do
         if [ -e \"\$artifact_path\" ]; then
             printf '%s\\n' \"\$artifact_path\"
         else
             printf 'Killarney optional path absent: %s\\n' \"\$artifact_path\" >&2
         fi
     done | tar -cf - -T -" \
    < "$MANIFEST" > "$KILLARNEY_ARCHIVE"

tar -xf "$KILLARNEY_ARCHIVE" -C "$STAGE_DIR"
poison_cache="$STAGE_DIR/ours_result/$ATTACK_RUN/poison_cache"
deltas=$(find "$poison_cache" -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null |
    wc -l | tr -d ' ')
[ "$deltas" -ge 7 ] || \
    die "Killarney archive contains only $deltas perturbations; expected at least 7"
printf 'Killarney download verified: %s defense perturbations.\n' "$deltas"

# Add the corrected local job and submit files to the same archive tree.
mkdir -p "$STAGE_DIR/sbatch" "$STAGE_DIR/defense_result" \
    "$STAGE_DIR/target_sets" "$STAGE_DIR/cache"
cp -R "$SCRIPT_DIR/sbatch/vulcan_remaining_20260827" "$STAGE_DIR/sbatch/"
cp "$SCRIPT_DIR/submit_vulcan-remaining.sh" \
   "$SCRIPT_DIR/submit_vulcan-remaining-defense.sh" \
   "$SCRIPT_DIR/submit_vulcan-remaining-augmentation.sh" \
   "$STAGE_DIR/"

if [ "$SUBMIT_AFTER_TRANSFER" = 1 ]; then
    remote_finish="sh submit_vulcan-remaining.sh"
else
    remote_finish="printf '\\nUpload complete. Run: cd $VULCAN_ROOT && sh submit_vulcan-remaining.sh\\n'"
fi

printf '\nAuthenticate to Vulcan once...\n'
# One Vulcan connection consumes the complete tar stream, installs it directly
# under PoisonBase, verifies the defense cache, and optionally submits all jobs.
tar -cf - -C "$STAGE_DIR" \
    ours_result defense_result target_sets cache sbatch \
    submit_vulcan-remaining.sh \
    submit_vulcan-remaining-defense.sh \
    submit_vulcan-remaining-augmentation.sh |
# Intentional splitting: SSH_OPTIONS contains separate OpenSSH arguments.
# shellcheck disable=SC2086
ssh $SSH_OPTIONS "$VULCAN_HOST" \
    "set -eu
     mkdir -p '$VULCAN_ROOT'
     cd '$VULCAN_ROOT'
     tar -xf -
     delta_count=\$(find 'ours_result/$ATTACK_RUN/poison_cache' -maxdepth 1 -type f -name 'delta_*.pt' | wc -l | tr -d '[:space:]')
     [ \"\$delta_count\" -ge 7 ] || {
         printf 'ERROR: Vulcan received only %s defense perturbations\\n' \"\$delta_count\" >&2
         exit 1
     }
     printf 'Vulcan upload verified: %s defense perturbations.\\n' \"\$delta_count\"
     $remote_finish"

printf '\nFinished with exactly one Killarney SSH connection and one Vulcan SSH connection.\n'
