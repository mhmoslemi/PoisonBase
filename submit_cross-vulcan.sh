#!/usr/bin/env bash
# Submit the 165 cross-architecture table cells moved from Killarney to Vulcan.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
JOB_ROOT="$SOURCE_ROOT/sbatch/cross_vulcan_moved"
LOG_ROOT="$SOURCE_ROOT/sbatch/logs"
EXPECTED_JOBS=165

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

job_count() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'cross_*.sh' | wc -l | tr -d ' '
}

ordered_jobs() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'cross_*.sh' -print |
    while IFS= read -r job; do
        requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
        [ -n "$requested" ] || die "missing #SBATCH --time in $job"
        seconds=$(printf '%s\n' "$requested" | awk -F '[-:]' \
            '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }')
        printf '%010d\t%s\n' "$seconds" "$job"
    done | sort -n -k1,1 -k2,2 | cut -f2-
}

line_count() {
    pattern="$1"
    file="$2"
    count=$(grep -c "$pattern" "$file" 2>/dev/null || true)
    printf '%s\n' "$count"
}

validate_jobs() {
    run_names=$(mktemp)
    job_names=$(mktemp)
    trap 'rm -f "$run_names" "$job_names"' EXIT HUP INT TERM

    find "$JOB_ROOT" -maxdepth 1 -type f -name 'cross_*.sh' -print |
    while IFS= read -r job; do
        [ "$(line_count '^#SBATCH --job-name=' "$job")" -eq 1 ] || \
            die "expected one job name in $job"
        [ "$(line_count '^#SBATCH --time=' "$job")" -eq 1 ] || \
            die "expected one requested time in $job"
        [ "$(line_count '^#SBATCH --cpus-per-task=1$' "$job")" -eq 1 ] || \
            die "job does not request exactly one CPU: $job"
        [ "$(line_count '^#SBATCH --mem=7G$' "$job")" -eq 1 ] || \
            die "job does not request 7 GB: $job"
        [ "$(line_count '^#SBATCH --gpus-per-node=l40s:1$' "$job")" -eq 1 ] || \
            die "job does not request one Vulcan L40S: $job"
        [ "$(line_count '^#SBATCH --mail' "$job")" -eq 0 ] || \
            die "email notification directive found in $job"
        [ "$(line_count '^export CROSS_RUN_NAME=' "$job")" -eq 1 ] || \
            die "expected one cross run in $job"
        [ "$(line_count '^export CROSS_NUM_TARGETS=5$' "$job")" -eq 1 ] || \
            die "cross cell does not use five targets: $job"
        [ "$(line_count '^export CROSS_NUM_VICTIMS=4$' "$job")" -eq 1 ] || \
            die "cross cell does not use four victims: $job"
        [ "$(line_count '^source /home/mmoslem3/scratch/PoisonBase/sbatch/_cross_job_common.sh$' "$job")" -eq 1 ] || \
            die "job does not use the Vulcan cross runtime: $job"
        sed -n 's/^export CROSS_RUN_NAME=//p' "$job" >> "$run_names"
        sed -n 's/^#SBATCH --job-name=//p' "$job" >> "$job_names"
    done

    [ "$(sort -u "$run_names" | wc -l | tr -d ' ')" -eq "$EXPECTED_JOBS" ] || \
        die 'duplicate CROSS_RUN_NAME values in moved jobs'
    [ "$(sort -u "$job_names" | wc -l | tr -d ' ')" -eq "$EXPECTED_JOBS" ] || \
        die 'duplicate Slurm job names in moved jobs'
    rm -f "$run_names" "$job_names"
    trap - EXIT HUP INT TERM
}

prepare_cifar10() {
    cifar=$(find "$SOURCE_ROOT/data" -type d -name cifar-10-batches-py -print -quit 2>/dev/null)
    [ -n "$cifar" ] && return 0
    [ -f "$ENV_ACTIVATE" ] || \
        die "Python environment activation script missing: $ENV_ACTIVATE"
    mkdir -p "$SOURCE_ROOT/data"
    printf 'Preparing CIFAR-10 under %s/data\n' "$SOURCE_ROOT"
    (
        . "$ENV_ACTIVATE"
        python - "$SOURCE_ROOT/data" <<'PY'
import sys
from torchvision.datasets import CIFAR10
CIFAR10(sys.argv[1], train=True, download=True)
CIFAR10(sys.argv[1], train=False, download=True)
PY
    )
}

verify_targets() {
    for model in ConvNetBN ResNet20BN VGG13BN; do
        for attack in fc gradmatch; do
            target="$SOURCE_ROOT/target_sets/xarch_${model}_${attack}_dog-bird_b0.005.json"
            [ -s "$target" ] || die "missing pinned target file: $target"
            TARGET_FILE="$target" python - <<'PY'
import json
import os

path = os.environ['TARGET_FILE']
with open(path) as handle:
    payload = json.load(handle)
count = len(payload.get('pairs', {}).get('dog-bird', {}).get('indices', []))
if count < 5:
    raise SystemExit('ERROR: %s has only %d pinned targets; need 5' % (path, count))
PY
        done
    done
}

cache_file_count() {
    directory="$1"
    if [ ! -d "$directory" ]; then printf '0\n'; return 0; fi
    find "$directory" -maxdepth 1 -type f -name 'net_*.pt' 2>/dev/null | \
        wc -l | tr -d ' '
}

verify_caches() {
    for model in ConvNetBN ResNet20BN VGG13BN; do
        surrogate="$SOURCE_ROOT/cache/surrogates/${model}_60ep_lr0.1_bs128_seed42"
        victim="$SOURCE_ROOT/cache/clean_victims/${model}_50ep_lr0.1_bs125_wd0_seed42"
        surrogate_count=$(cache_file_count "$surrogate")
        victim_count=$(cache_file_count "$victim")
        [ "$surrogate_count" -ge 20 ] || \
            die "$model cache has $surrogate_count/20 surrogates: $surrogate"
        [ "$victim_count" -ge 5 ] || \
            die "$model cache has $victim_count/5 victims: $victim"
        printf 'cache ready: %s (%s surrogates, %s victims)\n' \
            "$model" "$surrogate_count" "$victim_count"
    done
}

[ -d "$JOB_ROOT" ] || die "moved cross-job directory missing: $JOB_ROOT"
[ "$(job_count)" -eq "$EXPECTED_JOBS" ] || \
    die "expected $EXPECTED_JOBS moved cross jobs; found $(job_count)"
validate_jobs

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: %s validated one-cell jobs, shortest requested time first\n' \
        "$EXPECTED_JOBS"
    ordered_jobs | while IFS= read -r job; do
        printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"
    done
    exit 0
fi

case "$(hostname -s)" in
    vulcan*) ;;
    *) die "run this submitter on Vulcan (current host: $(hostname -s))" ;;
esac

mkdir -p "$LOG_ROOT"
prepare_cifar10
[ -f "$ENV_ACTIVATE" ] || \
    die "Python environment activation script missing: $ENV_ACTIVATE"
set +u
. "$ENV_ACTIVATE"
set -u
verify_targets
verify_caches

ordered=$(mktemp)
active_names=$(mktemp)
trap 'rm -f "$ordered" "$active_names"' EXIT HUP INT TERM
ordered_jobs > "$ordered"
squeue -h -u "${USER:-mmoslem3}" -o '%.200j' > "$active_names"

submitted=0
skipped=0
while IFS= read -r job; do
    job_name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    if grep -Fqx "$job_name" "$active_names"; then
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
    printf '%s\n' "$job_name" >> "$active_names"
    submitted=$((submitted + 1))
done < "$ordered"

printf 'Vulcan cross submission complete: %s submitted, %s active duplicates skipped\n' \
    "$submitted" "$skipped"
