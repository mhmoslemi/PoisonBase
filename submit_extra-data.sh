#!/usr/bin/env bash
# Build/check shared caches, expand the pinned target sets, and submit every
# pending Greedy cell in extra-data.tex on Vulcan. Result jobs are sorted by
# their actual #SBATCH --time request, shortest first. DRY_RUN=1 prints the
# ordered commands without submitting them.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export PERSIST_DATA_ROOT
JOB_ROOT="$SOURCE_ROOT/sbatch/extra_data"
PRECOMPUTE_ROOT="$SOURCE_ROOT/sbatch/extra_data_precompute"
EXTRA_DATA_MIN_INDEX="${EXTRA_DATA_MIN_INDEX:-1}"
EXTRA_DATA_MAX_INDEX="${EXTRA_DATA_MAX_INDEX:-54}"
mkdir -p "$SOURCE_ROOT/sbatch/logs"

if [ "${DRY_RUN:-0}" != 1 ]; then
    case "$(hostname -s)" in
        vulcan*) ;;
        *)
            echo "ERROR: submit_extra-data.sh must be run on vulcan.alliancecan.ca" >&2
            echo "Current host: $(hostname -s)" >&2
            exit 1
            ;;
    esac
fi

case "$EXTRA_DATA_MIN_INDEX:$EXTRA_DATA_MAX_INDEX" in
    *[!0-9:]*|:|*:)
        echo "ERROR: extra-data index bounds must be integers" >&2
        exit 1
        ;;
esac
[ "$EXTRA_DATA_MIN_INDEX" -ge 1 ] && \
[ "$EXTRA_DATA_MAX_INDEX" -le 54 ] && \
[ "$EXTRA_DATA_MIN_INDEX" -le "$EXTRA_DATA_MAX_INDEX" ] || {
    echo "ERROR: expected 1 <= EXTRA_DATA_MIN_INDEX <= EXTRA_DATA_MAX_INDEX <= 54" >&2
    exit 1
}

selected_jobs() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_*.sh' -print |
    while IFS= read -r job; do
        index=$(basename "$job" | cut -d_ -f2 | sed 's/^0*//')
        [ -n "$index" ] || index=0
        if [ "$index" -ge "$EXTRA_DATA_MIN_INDEX" ] && \
           [ "$index" -le "$EXTRA_DATA_MAX_INDEX" ]; then
            printf '%s\n' "$job"
        fi
    done
}

COUNT=$(selected_jobs | wc -l | tr -d ' ')
EXPECTED_COUNT=$((EXTRA_DATA_MAX_INDEX - EXTRA_DATA_MIN_INDEX + 1))
[ "$COUNT" -eq "$EXPECTED_COUNT" ] || {
    echo "expected $EXPECTED_COUNT one-cell jobs in index range $EXTRA_DATA_MIN_INDEX..$EXTRA_DATA_MAX_INDEX; found $COUNT" >&2
    exit 1
}

prepare_downloadable_inputs() {
    local train test cifar need_svhn need_cifar
    train=$(find "$PERSIST_DATA_ROOT" -type f -name train_32x32.mat -print -quit 2>/dev/null)
    test=$(find "$PERSIST_DATA_ROOT" -type f -name test_32x32.mat -print -quit 2>/dev/null)
    cifar=$(find "$PERSIST_DATA_ROOT" -type d -name cifar-100-python -print -quit 2>/dev/null)
    need_svhn=0
    need_cifar=0
    [ -n "$train" ] && [ -n "$test" ] || need_svhn=1
    [ -n "$cifar" ] || need_cifar=1
    [ "$need_svhn" -eq 1 ] || [ "$need_cifar" -eq 1 ] || return 0

    [ -f "$ENV_ACTIVATE" ] || {
        echo "ERROR: Python environment activation script missing: $ENV_ACTIVATE" >&2
        return 1
    }
    mkdir -p "$PERSIST_DATA_ROOT"
    echo "Preparing missing downloadable datasets under $PERSIST_DATA_ROOT"
    (
        . "$ENV_ACTIVATE"
        python - "$PERSIST_DATA_ROOT" "$need_svhn" "$need_cifar" <<'PY'
import os
import sys
from torchvision.datasets import CIFAR100, SVHN

root, need_svhn, need_cifar = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(root, exist_ok=True)
if need_svhn:
    print("downloading/checking SVHN train and test files", flush=True)
    SVHN(root, split="train", download=True)
    SVHN(root, split="test", download=True)
if need_cifar:
    print("downloading/checking CIFAR-100 train and test files", flush=True)
    CIFAR100(root, train=True, download=True)
    CIFAR100(root, train=False, download=True)
PY
    )
}

verify_inputs() {
    local train test cifar tiny
    train=$(find "$PERSIST_DATA_ROOT" -type f -name train_32x32.mat -print -quit 2>/dev/null)
    test=$(find "$PERSIST_DATA_ROOT" -type f -name test_32x32.mat -print -quit 2>/dev/null)
    cifar=$(find "$PERSIST_DATA_ROOT" -type d -name cifar-100-python -print -quit 2>/dev/null)
    tiny=$(find "$PERSIST_DATA_ROOT" -type f -name tinyimagenet.pt -print -quit 2>/dev/null)
    [ -n "$train" ] || {
        echo "ERROR: missing SVHN train_32x32.mat anywhere under $PERSIST_DATA_ROOT" >&2
        return 1
    }
    [ -n "$test" ] || {
        echo "ERROR: missing SVHN test_32x32.mat anywhere under $PERSIST_DATA_ROOT" >&2
        return 1
    }
    [ -n "$cifar" ] || {
        echo "ERROR: missing CIFAR-100 cifar-100-python directory under $PERSIST_DATA_ROOT" >&2
        return 1
    }
    [ -n "$tiny" ] || {
        echo "ERROR: missing Tiny ImageNet tinyimagenet.pt under $PERSIST_DATA_ROOT" >&2
        return 1
    }
}

submit_precompute() {
    local job output job_id
    job="$1"
    [ -f "$job" ] || {
        echo "ERROR: missing cache prerequisite script: $job" >&2
        return 1
    }
    output=$(sbatch --parsable --account="$ACCOUNT" "$job")
    job_id=${output%%;*}
    case "$job_id" in *[!0-9]*|'')
        echo "ERROR: could not parse job ID from sbatch output: $output" >&2
        return 1
        ;;
    esac
    echo "cache prerequisite: $(basename "$job") -> $job_id" >&2
    printf '%s\n' "$job_id"
}

cache_file_count() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        printf '0\n'
        return 0
    fi
    find "$dir" -maxdepth 1 -type f -name 'net_*.pt' 2>/dev/null | wc -l | tr -d ' '
}

ensure_precompute() {
    local job surrogate_cache victim_cache surrogate_count victim_count
    local job_name existing
    job="$1"
    surrogate_cache=$(sed -n 's/^export SURROGATE_CACHE=//p' "$job")
    victim_cache=$(sed -n 's/^export VICTIM_CACHE=//p' "$job")
    job_name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    [ -n "$surrogate_cache" ] && [ -n "$victim_cache" ] && [ -n "$job_name" ] || {
        echo "ERROR: incomplete cache metadata in $job" >&2
        return 1
    }
    surrogate_count=$(cache_file_count "$SOURCE_ROOT/cache/surrogates/$surrogate_cache")
    victim_count=$(cache_file_count "$SOURCE_ROOT/cache/clean_victims/$victim_cache")
    if [ "$surrogate_count" -ge 20 ] && [ "$victim_count" -ge 5 ]; then
        echo "cache prerequisite already satisfied: $job_name ($surrogate_count surrogates, $victim_count victims)" >&2
        return 0
    fi

    existing=$(squeue -h -u "${USER:-mmoslem3}" -n "$job_name" -o '%A' | sort -u | head -n 1)
    case "$existing" in
        '') submit_precompute "$job" ;;
        *[!0-9]*)
            echo "ERROR: invalid active job ID for $job_name: $existing" >&2
            return 1
            ;;
        *)
            echo "cache prerequisite: reusing active $job_name job $existing" >&2
            printf '%s\n' "$existing"
            ;;
    esac
}

submit_pin_targets() {
    local job dependency output job_id
    job="$1"
    dependency="$2"
    [ -f "$job" ] || {
        echo "ERROR: missing target-pinning prerequisite script: $job" >&2
        return 1
    }
    if [ -n "$dependency" ]; then
        output=$(sbatch --parsable --account="$ACCOUNT" \
            --dependency="afterok:$dependency" "$job")
    else
        output=$(sbatch --parsable --account="$ACCOUNT" "$job")
    fi
    job_id=${output%%;*}
    case "$job_id" in *[!0-9]*|'')
        echo "ERROR: could not parse job ID from sbatch output: $output" >&2
        return 1
        ;;
    esac
    echo "target prerequisite: $(basename "$job") -> $job_id" >&2
    printf '%s\n' "$job_id"
}

ordered_jobs() {
    selected_jobs |
    while IFS= read -r job; do
        requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
        [ -n "$requested" ] || {
            echo "missing #SBATCH --time in $job" >&2
            exit 1
        }
        seconds=$(printf '%s\n' "$requested" | awk -F '[-:]' \
            '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }')
        printf '%010d\t%s\n' "$seconds" "$job"
    done | sort -n -k1,1 -k2,2 | cut -f2-
}

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "sbatch --account=$ACCOUNT $PRECOMPUTE_ROOT/xdata_precompute_svhn.sh"
    echo "sbatch --account=$ACCOUNT $PRECOMPUTE_ROOT/xdata_precompute_cifar100.sh"
    echo "sbatch --account=$ACCOUNT $PRECOMPUTE_ROOT/xdata_precompute_tinyimagenet.sh"
    echo "sbatch --account=$ACCOUNT --dependency=afterok:PRECOMPUTE_SVHN $PRECOMPUTE_ROOT/xdata_pin_svhn.sh"
    echo "sbatch --account=$ACCOUNT --dependency=afterok:PRECOMPUTE_CIFAR100 $PRECOMPUTE_ROOT/xdata_pin_cifar100.sh"
    echo "sbatch --account=$ACCOUNT --dependency=afterok:PRECOMPUTE_TINYIMAGENET $PRECOMPUTE_ROOT/xdata_pin_tinyimagenet.sh"
    SVHN_DEP=PIN_SVHN
    CIFAR100_DEP=PIN_CIFAR100
    TINY_DEP=PIN_TINYIMAGENET
else
    prepare_downloadable_inputs
    verify_inputs
    SVHN_CACHE_DEP=$(ensure_precompute "$PRECOMPUTE_ROOT/xdata_precompute_svhn.sh")
    CIFAR100_CACHE_DEP=$(ensure_precompute "$PRECOMPUTE_ROOT/xdata_precompute_cifar100.sh")
    TINY_CACHE_DEP=$(ensure_precompute "$PRECOMPUTE_ROOT/xdata_precompute_tinyimagenet.sh")
    SVHN_DEP=$(submit_pin_targets "$PRECOMPUTE_ROOT/xdata_pin_svhn.sh" "$SVHN_CACHE_DEP")
    CIFAR100_DEP=$(submit_pin_targets "$PRECOMPUTE_ROOT/xdata_pin_cifar100.sh" "$CIFAR100_CACHE_DEP")
    TINY_DEP=$(submit_pin_targets "$PRECOMPUTE_ROOT/xdata_pin_tinyimagenet.sh" "$TINY_CACHE_DEP")
fi

echo "extra-data on Vulcan: $COUNT one-cell jobs ($EXTRA_DATA_MIN_INDEX..$EXTRA_DATA_MAX_INDEX), sorted by requested time"
ordered_jobs | while IFS= read -r job; do
    dataset=$(sed -n 's/^export DATASET=//p' "$job")
    case "$dataset" in
        SVHN) dependency=$SVHN_DEP ;;
        CIFAR100) dependency=$CIFAR100_DEP ;;
        TinyImageNet) dependency=$TINY_DEP ;;
        *)
            echo "ERROR: unsupported dataset in $job: $dataset" >&2
            exit 1
            ;;
    esac
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch --account=$ACCOUNT --dependency=afterok:$dependency $job"
    else
        sbatch --account="$ACCOUNT" --dependency="afterok:$dependency" "$job"
    fi
done
