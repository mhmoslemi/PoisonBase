#!/usr/bin/env bash
# Submit every pending Greedy cell in extra-data.tex on Vulcan. Jobs are sorted
# by their actual #SBATCH --time request, shortest first. DRY_RUN=1 prints the
# ordered commands without submitting them.

set -eu

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export PERSIST_DATA_ROOT
JOB_ROOT="$SOURCE_ROOT/sbatch/extra_data"
PRECOMPUTE_ROOT="$SOURCE_ROOT/sbatch/extra_data_precompute"
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

COUNT=$(find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_*.sh' -print | wc -l | tr -d ' ')
[ "$COUNT" -eq 54 ] || {
    echo "expected 54 one-cell jobs under $JOB_ROOT; found $COUNT" >&2
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

ordered_jobs() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'xdata_*.sh' -print |
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
    SVHN_DEP=PRECOMPUTE_SVHN
    CIFAR100_DEP=PRECOMPUTE_CIFAR100
    TINY_DEP=PRECOMPUTE_TINYIMAGENET
    echo "sbatch --account=$ACCOUNT $PRECOMPUTE_ROOT/xdata_precompute_svhn.sh"
    echo "sbatch --account=$ACCOUNT $PRECOMPUTE_ROOT/xdata_precompute_cifar100.sh"
    echo "sbatch --account=$ACCOUNT $PRECOMPUTE_ROOT/xdata_precompute_tinyimagenet.sh"
else
    prepare_downloadable_inputs
    verify_inputs
    SVHN_DEP=$(submit_precompute "$PRECOMPUTE_ROOT/xdata_precompute_svhn.sh")
    CIFAR100_DEP=$(submit_precompute "$PRECOMPUTE_ROOT/xdata_precompute_cifar100.sh")
    TINY_DEP=$(submit_precompute "$PRECOMPUTE_ROOT/xdata_precompute_tinyimagenet.sh")
fi

echo "extra-data on Vulcan: $COUNT one-cell jobs, sorted by requested time"
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
