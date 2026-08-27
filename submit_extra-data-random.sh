#!/usr/bin/env bash
# Submit the 54 Random-selection cells added to the broader extra-data table.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
PYTHON_BIN="${PYTHON_BIN:-/home/mmoslem3/ENV/bin/python}"
JOB_ROOT="$SOURCE_ROOT/sbatch/extra_data_random"
EXPECTED_JOBS=54

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

line_count() {
    pattern="$1"
    file="$2"
    grep -c "$pattern" "$file" 2>/dev/null || true
}

ordered_jobs() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_r*_random.sh' -print |
    while IFS= read -r job; do
        requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
        [ -n "$requested" ] || die "missing #SBATCH --time in $job"
        seconds=$(printf '%s\n' "$requested" | awk -F '[-:]' \
            '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }')
        printf '%010d\t%s\n' "$seconds" "$job"
    done | sort -n -k1,1 -k2,2 | cut -f2-
}

validate_jobs() {
    names=$(mktemp)
    trap 'rm -f "$names"' EXIT HUP INT TERM
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_r*_random.sh' -print |
    while IFS= read -r job; do
        [ "$(line_count '^#SBATCH --job-name=' "$job")" -eq 1 ] || \
            die "expected exactly one Slurm job name in $job"
        [ "$(line_count '^#SBATCH --time=' "$job")" -eq 1 ] || \
            die "expected exactly one requested time in $job"
        [ "$(line_count '^#SBATCH --cpus-per-task=1$' "$job")" -eq 1 ] || \
            die "job does not request exactly one CPU: $job"
        [ "$(line_count '^#SBATCH --gpus-per-node=l40s:1$' "$job")" -eq 1 ] || \
            die "job does not request exactly one Vulcan L40S: $job"
        [ "$(line_count '^#SBATCH --mail' "$job")" -eq 0 ] || \
            die "email directive found in $job"
        [ "$(line_count '^export SELECTION=random$' "$job")" -eq 1 ] || \
            die "job is not a single Random-selection cell: $job"
        [ "$(line_count '^export DATASET=' "$job")" -eq 1 ] || \
            die "expected exactly one dataset in $job"
        [ "$(line_count '^export ATTACK=' "$job")" -eq 1 ] || \
            die "expected exactly one attack in $job"
        [ "$(line_count '^source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_job_common.sh$' "$job")" -eq 1 ] || \
            die "job does not use the Vulcan extra-data runtime: $job"
        sed -n 's/^#SBATCH --job-name=//p' "$job" >> "$names"
    done
    [ "$(sort -u "$names" | wc -l | tr -d ' ')" -eq "$EXPECTED_JOBS" ] || \
        die 'duplicate Slurm job names in Random-selection jobs'
    rm -f "$names"
    trap - EXIT HUP INT TERM
}

cache_file_count() {
    directory="$1"
    if [ ! -d "$directory" ]; then printf '0\n'; return 0; fi
    find "$directory" -maxdepth 1 -type f -name 'net_*.pt' 2>/dev/null | \
        wc -l | tr -d ' '
}

verify_data_and_caches() {
    train=$(find "$SOURCE_ROOT/data" -type f -name train_32x32.mat -print -quit 2>/dev/null)
    test=$(find "$SOURCE_ROOT/data" -type f -name test_32x32.mat -print -quit 2>/dev/null)
    cifar=$(find "$SOURCE_ROOT/data" -type d -name cifar-100-python -print -quit 2>/dev/null)
    tiny=$(find "$SOURCE_ROOT/data" -type f -name tinyimagenet.pt -print -quit 2>/dev/null)
    [ -n "$train" ] && [ -n "$test" ] || die 'SVHN input is incomplete'
    [ -n "$cifar" ] || die 'CIFAR-100 input is missing'
    [ -n "$tiny" ] || die 'Tiny ImageNet input is missing'

    for spec in \
        'SVHN ConvNetBN 5 0.01' \
        'CIFAR100 ResNet18BN 4 0.1' \
        'TinyImageNet ResNet18BN 4 0.1'; do
        set -- $spec
        dataset=$1
        model=$2
        victims=$3
        victim_lr=$4
        surrogate="$SOURCE_ROOT/cache/surrogates/${dataset}_${model}_60ep_lr0.1_bs128_seed42"
        victim="$SOURCE_ROOT/cache/clean_victims/${dataset}_${model}_50ep_lr${victim_lr}_bs125_wd0_seed42"
        surrogate_count=$(cache_file_count "$surrogate")
        victim_count=$(cache_file_count "$victim")
        [ "$surrogate_count" -ge 20 ] || \
            die "$dataset cache has $surrogate_count/20 surrogates"
        [ "$victim_count" -ge "$victims" ] || \
            die "$dataset cache has $victim_count/$victims victims"
        printf 'cache ready: %s/%s (%s surrogates, %s victims)\n' \
            "$dataset" "$model" "$surrogate_count" "$victim_count"
    done
}

verify_targets() {
    [ -x "$PYTHON_BIN" ] || die "Python executable missing: $PYTHON_BIN"
    "$PYTHON_BIN" - "$JOB_ROOT" <<'PY'
import glob
import json
import os
import re
import sys

root = sys.argv[1]
jobs = sorted(glob.glob(os.path.join(root, 'xdata_r*_random.sh')))
for job in jobs:
    text = open(job).read()
    fields = {}
    for key in ('DATASET', 'CLASS_PAIR', 'TARGET_FILE'):
        match = re.search(r'^export %s=(.+)$' % key, text, re.MULTILINE)
        if not match:
            raise SystemExit('ERROR: missing %s in %s' % (key, job))
        fields[key] = match.group(1)
    needed = 6 if fields['DATASET'] == 'SVHN' else 4
    path = os.path.join(os.path.dirname(os.path.dirname(root)), fields['TARGET_FILE'])
    with open(path) as handle:
        payload = json.load(handle)
    indices = payload.get('pairs', {}).get(fields['CLASS_PAIR'], {}).get('indices', [])
    if len(indices) < needed:
        raise SystemExit(
            'ERROR: %s has %d targets for %s; need %d'
            % (path, len(indices), fields['CLASS_PAIR'], needed)
        )
print('target sets ready for %d Random-selection cells' % len(jobs))
PY
}

[ -d "$JOB_ROOT" ] || die "Random-selection job directory missing: $JOB_ROOT"
count=$(find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_r*_random.sh' | \
    wc -l | tr -d ' ')
[ "$count" -eq "$EXPECTED_JOBS" ] || \
    die "expected $EXPECTED_JOBS Random-selection jobs; found $count"
validate_jobs

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: %s Random-selection cells, shortest requested time first\n' \
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

mkdir -p "$SOURCE_ROOT/sbatch/logs"
verify_data_and_caches
verify_targets

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

printf 'Vulcan extra-data Random submission complete: %s submitted, %s active duplicates skipped\n' \
    "$submitted" "$skipped"
