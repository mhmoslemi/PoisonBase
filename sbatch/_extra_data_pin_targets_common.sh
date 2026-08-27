#!/usr/bin/env bash
# Pin the dataset-specific target set after the shared clean-victim cache is
# ready. Dataset array scripts define PAIRS/TARGET_FILES and source this file.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
RUN_ROOT="${SLURM_TMPDIR:-}/extra_data_pin"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
LOCAL_CACHE_ROOT="$RUN_ROOT/cache"

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
        CIFAR100)
            src="$(find "$PERSIST_DATA_ROOT" -type d -name cifar-100-python -print -quit 2>/dev/null)"
            [ -n "$src" ] || die "CIFAR-100 input missing under $PERSIST_DATA_ROOT"
            rsync -a "$src" "$LOCAL_DATA_ROOT/"
            ;;
        SVHN)
            train_src="$(find "$PERSIST_DATA_ROOT" -type f -name train_32x32.mat -print -quit 2>/dev/null)"
            test_src="$(find "$PERSIST_DATA_ROOT" -type f -name test_32x32.mat -print -quit 2>/dev/null)"
            [ -n "$train_src" ] || die "SVHN train input missing under $PERSIST_DATA_ROOT"
            [ -n "$test_src" ] || die "SVHN test input missing under $PERSIST_DATA_ROOT"
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

target_count() {
    python - "$1" "$2" <<'PY'
import json
import os
import sys

path, pair = sys.argv[1:]
if not os.path.isfile(path):
    print(0)
else:
    try:
        with open(path) as handle:
            payload = json.load(handle)
        print(len(payload.get("pairs", {}).get(pair, {}).get("indices", [])))
    except (OSError, ValueError, TypeError):
        print(0)
PY
}

main() {
    local task pair target_file persistent_target local_target count cache_count
    for required in DATASET MODEL NUM_TARGETS NUM_VICTIMS VICTIM_CACHE VICTIM_LR; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done
    [ -n "${SLURM_TMPDIR:-}" ] || die 'SLURM_TMPDIR is unset; submit this file with sbatch'
    task="${SLURM_ARRAY_TASK_ID:-}"
    [ -n "$task" ] || die 'SLURM_ARRAY_TASK_ID is unset'
    pair="${PAIRS[$task]:-}"
    target_file="${TARGET_FILES[$task]:-}"
    [ -n "$pair" ] || die "no pair for array task $task"
    [ -n "$target_file" ] || die "no target file for array task $task"

    module load python/3.11.5 cuda/12.6 cudnn
    [ -f "$ENV_ACTIVATE" ] || die "Python environment activation script missing: $ENV_ACTIVATE"
    source "$ENV_ACTIVATE"

    persistent_target="$SOURCE_ROOT/$target_file"
    count="$(target_count "$persistent_target" "$pair")"
    if [ "$count" -ge "$NUM_TARGETS" ]; then
        say "target prerequisite already satisfied: $persistent_target has $count target(s)"
        exit 0
    fi

    stage_dataset
    mkdir -p "$RUN_ROOT/appendix" "$LOCAL_CACHE_ROOT/clean_victims" \
        "$RUN_ROOT/target_sets"
    for file in final_update.py networks.py utils.py appendix/pin_targets.py; do
        [ -f "$SOURCE_ROOT/$file" ] || die "required source file missing: $SOURCE_ROOT/$file"
        mkdir -p "$RUN_ROOT/$(dirname "$file")"
        rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/$file"
    done
    copy_dir_if_present "$SOURCE_ROOT/cache/clean_victims/$VICTIM_CACHE" \
        "$LOCAL_CACHE_ROOT/clean_victims/$VICTIM_CACHE"
    cache_count="$(find "$LOCAL_CACHE_ROOT/clean_victims/$VICTIM_CACHE" \
        -maxdepth 1 -type f -name 'net_*.pt' 2>/dev/null | wc -l | tr -d ' ')"
    [ "$cache_count" -ge "$NUM_VICTIMS" ] || \
        die "need $NUM_VICTIMS cached clean victims in $VICTIM_CACHE; found $cache_count"

    local_target="$RUN_ROOT/target_sets/$(basename "$target_file")"
    say "pinning: dataset=$DATASET model=$MODEL pair=$pair targets=$NUM_TARGETS victims=$NUM_VICTIMS"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'
    srun --ntasks=1 python "$RUN_ROOT/appendix/pin_targets.py" \
        --dataset "$DATASET" --data_path "$LOCAL_DATA_ROOT" \
        --cache_dir "$LOCAL_CACHE_ROOT" --model "$MODEL" --pair "$pair" \
        --target_select random --num_targets "$NUM_TARGETS" \
        --num_victims "$NUM_VICTIMS" --victim_epochs 50 \
        --victim_lr "$VICTIM_LR" --victim_bs 125 --victim_decay 40 \
        --victim_wd 0.0 --seed 42 --out "$local_target" --force

    count="$(target_count "$local_target" "$pair")"
    [ "$count" -ge "$NUM_TARGETS" ] || \
        die "pinning produced only $count target(s), expected $NUM_TARGETS"
    mkdir -p "$(dirname "$persistent_target")"
    rsync -a "$local_target" "$persistent_target"
    say "sync: $count pinned targets -> $persistent_target"
}

main "$@"
