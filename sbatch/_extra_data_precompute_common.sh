#!/usr/bin/env bash
# Build one member of an extra-dataset surrogate/victim cache. The three Slurm
# array scripts set the dataset-specific variables and one array task owns each
# cache index, so concurrent tasks never train the same checkpoint.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
RUN_ROOT="${SLURM_TMPDIR:-}/extra_data_precompute"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
LOCAL_CACHE_ROOT="$RUN_ROOT/cache"
STEP_PID=""
SYNCED=0

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

copy_dir_if_present() {
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    rsync -a --exclude='.lock' --exclude='*.tmp' "$src/" "$dst/"
}

stage_dataset() {
    local src train_src test_src
    mkdir -p "$LOCAL_DATA_ROOT"
    case "$DATASET" in
        CIFAR10)
            src="$(find "$PERSIST_DATA_ROOT" -type d -name cifar-10-batches-py -print -quit 2>/dev/null)"
            [ -n "$src" ] || die "CIFAR-10 input missing under $PERSIST_DATA_ROOT"
            rsync -a "$src" "$LOCAL_DATA_ROOT/"
            ;;
        CIFAR100)
            src="$(find "$PERSIST_DATA_ROOT" -type d -name cifar-100-python -print -quit 2>/dev/null)"
            [ -n "$src" ] || die "CIFAR-100 input missing under $PERSIST_DATA_ROOT"
            rsync -a "$src" "$LOCAL_DATA_ROOT/"
            ;;
        SVHN)
            train_src="$(find "$PERSIST_DATA_ROOT" -type f -name train_32x32.mat -print -quit 2>/dev/null)"
            test_src="$(find "$PERSIST_DATA_ROOT" -type f -name test_32x32.mat -print -quit 2>/dev/null)"
            [ -n "$train_src" ] || die "SVHN input missing: train_32x32.mat under $PERSIST_DATA_ROOT"
            [ -n "$test_src" ] || die "SVHN input missing: test_32x32.mat under $PERSIST_DATA_ROOT"
            rsync -a "$train_src" "$LOCAL_DATA_ROOT/train_32x32.mat"
            rsync -a "$test_src" "$LOCAL_DATA_ROOT/test_32x32.mat"
            ;;
        TinyImageNet)
            src="$(find "$PERSIST_DATA_ROOT" -type f -name tinyimagenet.pt -print -quit 2>/dev/null)"
            [ -n "$src" ] || die "Tiny ImageNet input missing under $PERSIST_DATA_ROOT"
            rsync -a "$src" "$LOCAL_DATA_ROOT/tinyimagenet.pt"
            ;;
        *) die "unsupported extra-data dataset: $DATASET" ;;
    esac
}

stage_code_and_caches() {
    local file
    mkdir -p "$RUN_ROOT" "$LOCAL_CACHE_ROOT/surrogates" \
        "$LOCAL_CACHE_ROOT/clean_victims"
    for file in final_update.py networks.py utils.py; do
        [ -f "$SOURCE_ROOT/$file" ] || die "required source file missing: $SOURCE_ROOT/$file"
        rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
    done
    copy_dir_if_present "$SOURCE_ROOT/cache/surrogates/$SURROGATE_CACHE" \
        "$LOCAL_CACHE_ROOT/surrogates/$SURROGATE_CACHE"
    copy_dir_if_present "$SOURCE_ROOT/cache/clean_victims/$VICTIM_CACHE" \
        "$LOCAL_CACHE_ROOT/clean_victims/$VICTIM_CACHE"
}

sync_cache_dir() {
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    rsync -a --ignore-existing --exclude='*.tmp' "$src/" "$dst/"
}

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    sync_cache_dir "$LOCAL_CACHE_ROOT/surrogates/$SURROGATE_CACHE" \
        "$SOURCE_ROOT/cache/surrogates/$SURROGATE_CACHE"
    sync_cache_dir "$LOCAL_CACHE_ROOT/clean_victims/$VICTIM_CACHE" \
        "$SOURCE_ROOT/cache/clean_victims/$VICTIM_CACHE"
    say "sync: cache member $PRECOMPUTE_ID -> $SOURCE_ROOT/cache ($DATASET/$MODEL)"
}

handle_signal() {
    local signal="$1"
    say "signal: received $signal; stopping cache training before final sync"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

main() {
    local required part status
    for required in DATASET MODEL SURROGATE_CACHE VICTIM_CACHE VICTIM_LR; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done
    [ -n "${SLURM_TMPDIR:-}" ] || die 'SLURM_TMPDIR is unset; submit this file with sbatch'
    PRECOMPUTE_ID="${SLURM_ARRAY_TASK_ID:-${PRECOMPUTE_ID:-}}"
    [ -n "$PRECOMPUTE_ID" ] || die 'SLURM_ARRAY_TASK_ID/PRECOMPUTE_ID is unset'
    case "$PRECOMPUTE_ID" in *[!0-9]*|'') die "invalid cache index: $PRECOMPUTE_ID" ;; esac
    [ "$PRECOMPUTE_ID" -lt 20 ] || die "cache index must be in 0..19: $PRECOMPUTE_ID"
    if [ "$PRECOMPUTE_ID" -lt 5 ]; then part=both; else part=surrogate; fi

    module load python/3.11.5 cuda/12.6 cudnn
    [ -f "$ENV_ACTIVATE" ] || die "Python environment activation script missing: $ENV_ACTIVATE"
    source "$ENV_ACTIVATE"

    trap 'handle_signal USR1' USR1
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap sync_outputs EXIT

    stage_dataset
    stage_code_and_caches
    say "cache: dataset=$DATASET model=$MODEL id=$PRECOMPUTE_ID part=$part"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    srun --ntasks=1 python "$RUN_ROOT/final_update.py" \
        --dataset "$DATASET" --data_path "$LOCAL_DATA_ROOT" --seed 42 \
        --cache_dir "$LOCAL_CACHE_ROOT" --out_dir "$RUN_ROOT/ours_result" \
        --model "$MODEL" --gpus all \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
        --surrogate_bs 128 --surrogate_decay 35 45 \
        --num_victims 5 --victim_epochs 50 --victim_lr "$VICTIM_LR" \
        --victim_bs 125 --victim_decay 40 --victim_wd 0.0 \
        --precompute_only --precompute_part "$part" \
        --precompute_id "$PRECOMPUTE_ID" &
    STEP_PID=$!
    set +e
    wait "$STEP_PID"
    status=$?
    set -e
    STEP_PID=""
    sync_outputs
    trap - EXIT
    exit "$status"
}

main "$@"
