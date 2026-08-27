#!/usr/bin/env bash
# Submit exactly the Killarney jobs moved in the 2026-08-26 queue snapshot:
# extra-data cells 017..054 and cross-architecture cells 189..208.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
PRECOMPUTE_ROOT="$SOURCE_ROOT/sbatch/cross_vulcan_precompute"
CROSS_ROOT="$SOURCE_ROOT/sbatch/cross_expanded"
LOG_ROOT="$SOURCE_ROOT/sbatch/logs"
mkdir -p "$LOG_ROOT"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [ "${DRY_RUN:-0}" != 1 ]; then
    case "$(hostname -s)" in
        vulcan*) ;;
        *) die "submit_vulcan_moved.sh must be run on vulcan.alliancecan.ca (current host: $(hostname -s))" ;;
    esac
fi

cross_jobs() {
    find "$CROSS_ROOT" -maxdepth 1 -type f -name 'cross_*.sh' -print |
    while IFS= read -r job; do
        index=$(basename "$job" | cut -d_ -f2 | sed 's/^0*//')
        [ -n "$index" ] || index=0
        if [ "$index" -ge 189 ] && [ "$index" -le 208 ]; then
            printf '%s\n' "$job"
        fi
    done
}

ordered_cross_jobs() {
    cross_jobs |
    while IFS= read -r job; do
        requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
        [ -n "$requested" ] || die "missing #SBATCH --time in $job"
        seconds=$(printf '%s\n' "$requested" | awk -F '[-:]' \
            '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }')
        printf '%010d\t%s\n' "$seconds" "$job"
    done | sort -n -k1,1 -k2,2 | cut -f2-
}

[ "$(cross_jobs | wc -l | tr -d ' ')" -eq 20 ] || \
    die "expected exactly 20 cross jobs (189..208)"

prepare_cifar10() {
    local cifar
    cifar=$(find "$SOURCE_ROOT/data" -type d -name cifar-10-batches-py -print -quit 2>/dev/null)
    [ -n "$cifar" ] && return 0
    [ -f "$ENV_ACTIVATE" ] || die "Python environment activation script missing: $ENV_ACTIVATE"
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

verify_cross_targets() {
    local model target
    for model in ConvNetBN ResNet20BN VGG13BN; do
        target="$SOURCE_ROOT/target_sets/xarch_${model}_gradmatch_dog-bird_b0.005.json"
        [ -s "$target" ] || die "missing pinned cross target file: $target"
        TARGET_FILE="$target" python - <<'PY'
import json
import os
path = os.environ['TARGET_FILE']
with open(path) as handle:
    payload = json.load(handle)
count = len(payload.get('pairs', {}).get('dog-bird', {}).get('indices', []))
if count < 5:
    raise SystemExit('ERROR: %s has only %d pinned target(s); need 5' % (path, count))
PY
    done
}

cache_file_count() {
    local dir="$1"
    if [ ! -d "$dir" ]; then printf '0\n'; return 0; fi
    find "$dir" -maxdepth 1 -type f -name 'net_*.pt' 2>/dev/null | wc -l | tr -d ' '
}

ensure_cross_cache() {
    local job surrogate_cache victim_cache job_name s_count v_count output job_id existing
    job="$1"
    surrogate_cache=$(sed -n 's/^export SURROGATE_CACHE=//p' "$job")
    victim_cache=$(sed -n 's/^export VICTIM_CACHE=//p' "$job")
    job_name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    [ -n "$surrogate_cache" ] && [ -n "$victim_cache" ] && [ -n "$job_name" ] || \
        die "incomplete cache metadata in $job"
    s_count=$(cache_file_count "$SOURCE_ROOT/cache/surrogates/$surrogate_cache")
    v_count=$(cache_file_count "$SOURCE_ROOT/cache/clean_victims/$victim_cache")
    if [ "$s_count" -ge 20 ] && [ "$v_count" -ge 5 ]; then
        printf 'cross cache ready: %s (%s surrogates, %s victims)\n' "$job_name" "$s_count" "$v_count" >&2
        return 0
    fi

    existing=$(squeue -h -u "${USER:-mmoslem3}" -n "$job_name" -o '%A' | sort -u | head -n 1)
    if [ -n "$existing" ]; then
        case "$existing" in *[!0-9]*) die "invalid active job ID for $job_name: $existing" ;; esac
        printf 'cross cache: reusing active %s job %s\n' "$job_name" "$existing" >&2
        printf '%s\n' "$existing"
        return 0
    fi

    output=$(sbatch --parsable --account="$ACCOUNT" "$job")
    job_id=${output%%;*}
    case "$job_id" in *[!0-9]*|'') die "could not parse job ID from sbatch output: $output" ;; esac
    printf 'cross cache: %s -> %s\n' "$(basename "$job")" "$job_id" >&2
    printf '%s\n' "$job_id"
}

submit_cross_jobs() {
    local dependencies="" dep job
    for job in \
        "$PRECOMPUTE_ROOT/cross_precompute_convnet.sh" \
        "$PRECOMPUTE_ROOT/cross_precompute_resnet20.sh" \
        "$PRECOMPUTE_ROOT/cross_precompute_vgg13.sh"; do
        [ -f "$job" ] || die "missing cross cache prerequisite: $job"
        dep=$(ensure_cross_cache "$job")
        [ -z "$dep" ] || dependencies="${dependencies:+$dependencies:}$dep"
    done
    printf 'cross architecture on Vulcan: 20 one-cell jobs (189..208), shortest first\n'
    ordered_cross_jobs | while IFS= read -r job; do
        if [ -n "$dependencies" ]; then
            sbatch --account="$ACCOUNT" --dependency="afterok:$dependencies" "$job"
        else
            sbatch --account="$ACCOUNT" "$job"
        fi
    done
}

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: cross cache prerequisites (submitted only if missing)\n'
    for job in "$PRECOMPUTE_ROOT"/*.sh; do
        printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"
    done
    ordered_cross_jobs | while IFS= read -r job; do
        printf 'sbatch --account=%s --dependency=afterok:CROSS_CACHE_JOBS %s\n' "$ACCOUNT" "$job"
    done
    EXTRA_DATA_MIN_INDEX=17 EXTRA_DATA_MAX_INDEX=54 \
        DRY_RUN=1 sh "$SOURCE_ROOT/submit_extra-data.sh"
    exit 0
fi

prepare_cifar10
. "$ENV_ACTIVATE"
verify_cross_targets
submit_cross_jobs

EXTRA_DATA_MIN_INDEX=17 EXTRA_DATA_MAX_INDEX=54 \
    sh "$SOURCE_ROOT/submit_extra-data.sh"

printf 'submitted the 58 moved one-cell jobs on Vulcan\n'
