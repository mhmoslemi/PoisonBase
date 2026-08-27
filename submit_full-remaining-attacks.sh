#!/bin/sh
# Submit the eight still-blank full.tex attack cells. The sbatch files mirror
# sbatch/attack/attack_007_convnet_gradmatch_dog_bird_b0_005_ours_j.sh and
# use the /home/mmoslem3/scratch/attack_if cluster layout. DRY_RUN=1 validates
# and prints the ordered commands without submitting them.

set -eu

ROOT="${ROOT:-/home/mmoslem3/scratch/attack_if}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
JOB_ROOT="$ROOT/sbatch/full_remaining_attack007"
LOG_ROOT="$ROOT/sbatch/logs2"
EXPECTED_JOBS=8
JOB_LIST=""
ORDERED=""
ACTIVE_NAMES=""
JOB_NAMES=""

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
    for path in "$JOB_LIST" "$ORDERED" "$ACTIVE_NAMES" "$JOB_NAMES"; do
        [ -z "$path" ] || [ ! -e "$path" ] || unlink "$path"
    done
}
trap cleanup EXIT HUP INT TERM

line_count() {
    pattern="$1"
    file="$2"
    count=$(grep -c "$pattern" "$file" 2>/dev/null || true)
    printf '%s\n' "$count"
}

requested_seconds() {
    printf '%s\n' "$1" | awk -F '[-:]' \
        '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }'
}

JOB_LIST=$(mktemp)
[ -d "$JOB_ROOT" ] || die "job directory missing: $JOB_ROOT"
find "$JOB_ROOT" -maxdepth 1 -type f -name 'b2040_attack_*.sh' -print | \
    sort > "$JOB_LIST"

count=$(wc -l < "$JOB_LIST" | tr -d ' ')
[ "$count" -eq "$EXPECTED_JOBS" ] || \
    die "expected $EXPECTED_JOBS full-table attack jobs; found $count"

JOB_NAMES=$(mktemp)
while IFS= read -r job; do
    [ "$(line_count '^#SBATCH --job-name=' "$job")" -eq 1 ] || \
        die "expected one Slurm job name in $job"
    [ "$(line_count '^#SBATCH --time=' "$job")" -eq 1 ] || \
        die "expected one requested time in $job"
    [ "$(line_count '^#SBATCH --cpus-per-task=1$' "$job")" -eq 1 ] || \
        die "job does not request exactly one CPU: $job"
    [ "$(line_count '^#SBATCH --mem=7G$' "$job")" -eq 1 ] || \
        die "job does not request 7 GB: $job"
    [ "$(line_count '^#SBATCH --gres=gpu:l40s:1$' "$job")" -eq 1 ] || \
        die "job does not match the attack_007 L40S directive: $job"
    [ "$(line_count '^#SBATCH --mail' "$job")" -eq 0 ] || \
        die "email directive found in $job"
    [ "$(line_count '^#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs2/' "$job")" -eq 1 ] || \
        die "job output is not routed to attack_if/sbatch/logs2: $job"
    [ "$(line_count '^export JOB_KIND=attack$' "$job")" -eq 1 ] || \
        die "job is not exactly one attack cell: $job"
    [ "$(line_count '^export BUDGETS=[^[:space:]]\+$' "$job")" -eq 1 ] || \
        die "job does not specify exactly one budget: $job"
    [ "$(line_count '^export NUM_TARGETS=8$' "$job")" -eq 1 ] || \
        die "job does not use eight targets: $job"
    [ "$(line_count '^export NUM_VICTIMS=6$' "$job")" -eq 1 ] || \
        die "job does not use six victims: $job"
    [ "$(line_count '^source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh$' "$job")" -eq 1 ] || \
        die "job does not use the attack_007 runtime path: $job"
    sed -n 's/^#SBATCH --job-name=//p' "$job" >> "$JOB_NAMES"
done < "$JOB_LIST"

[ "$(sort -u "$JOB_NAMES" | wc -l | tr -d ' ')" -eq "$EXPECTED_JOBS" ] || \
    die "duplicate Slurm job names in $JOB_ROOT"
unlink "$JOB_NAMES"
JOB_NAMES=""

ORDERED=$(mktemp)
while IFS= read -r job; do
    requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
    seconds=$(requested_seconds "$requested")
    printf '%010d\t%s\n' "$seconds" "$job"
done < "$JOB_LIST" | sort -n -k1,1 -k2,2 | cut -f2- > "$ORDERED"

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: %s attack_007-style one-cell jobs validated; shortest first\n' \
        "$EXPECTED_JOBS"
    while IFS= read -r job; do
        printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"
    done < "$ORDERED"
    exit 0
fi

[ -f "$ROOT/sbatch/_job_common.sh" ] || \
    die "missing shared runtime: $ROOT/sbatch/_job_common.sh"
# [ -d "$ROOT/data/cifar-10-batches-py" ] || \
#     die "CIFAR-10 input missing under $ROOT/data"
mkdir -p "$LOG_ROOT"

ACTIVE_NAMES=$(mktemp)
squeue -h -u "${USER:-mmoslem3}" -o '%.200j' > "$ACTIVE_NAMES"
submitted=0
skipped=0

while IFS= read -r job; do
    job_name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    if grep -Fqx "$job_name" "$ACTIVE_NAMES"; then
        printf 'skip active duplicate: %s\n' "$job_name"
        skipped=$((skipped + 1))
        continue
    fi
    output=$(sbatch --parsable --account="$ACCOUNT" "$job")
    job_id=${output%%;*}
    case "$job_id" in
        *[!0-9]*|'') die "could not parse job ID from sbatch output: $output" ;;
    esac
    printf 'submitted %s -> %s\n' "$job_name" "$job_id"
    printf '%s\n' "$job_name" >> "$ACTIVE_NAMES"
    submitted=$((submitted + 1))
done < "$ORDERED"

printf 'Full-table attack submission complete: %s submitted, %s active duplicates skipped\n' \
    "$submitted" "$skipped"
