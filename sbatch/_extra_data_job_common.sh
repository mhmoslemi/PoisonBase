#!/usr/bin/env bash
# Shared runtime for the one-cell jobs that fill tab:cross-dataset-greedy in
# extra-data.tex. Each generated sbatch file exports exactly one dataset/model/
# pair/budget/attack configuration before sourcing this file.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-/home/mmoslem3/scratch/PoisonBase/data}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
RUN_ROOT="${SLURM_TMPDIR:-}/extra_data_if"
LOCAL_DATA_ROOT="${SLURM_TMPDIR:-}/extra_data"
PERSIST_RESULT_ROOT="${PERSIST_RESULT_ROOT:-$SOURCE_ROOT/extra_data_result}"
LOCAL_RESULT_ROOT="$RUN_ROOT/extra_data_result"
STEP_PID=""
SYNCED=0

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

fmt_g() {
    python - "$1" <<'PY'
import sys
print('%g' % float(sys.argv[1]))
PY
}

run_name() {
    local budget name
    budget="$(fmt_g "$BUDGET")"
    name="${DATASET}_${MODEL}_${ATTACK}_ours_${CLASS_PAIR}_b${budget}_eps8_seed42_lam1_cosine"
    [ "$ATTACK" = sapa ] && name+="_worst0.05"
    name+="_ce5"
    printf '%s\n' "$name"
}

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

cache_file_count() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        printf '0\n'
        return 0
    fi
    find "$dir" -maxdepth 1 -type f -name 'net_*.pt' 2>/dev/null | wc -l | tr -d ' '
}

stage_code_target_and_caches() {
    local file src target_src s_count v_count
    mkdir -p "$RUN_ROOT" "$RUN_ROOT/cache/surrogates" \
        "$RUN_ROOT/cache/clean_victims" "$LOCAL_RESULT_ROOT" \
        "$RUN_ROOT/target_sets"

    for file in final_update.py networks.py utils.py; do
        [ -f "$SOURCE_ROOT/$file" ] || die "required source file missing: $SOURCE_ROOT/$file"
        rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
    done

    target_src="$SOURCE_ROOT/$TARGET_FILE"
    [ -s "$target_src" ] || die "pinned target file missing: $target_src"
    rsync -a "$target_src" "$RUN_ROOT/target_sets/"

    src="$SOURCE_ROOT/cache/surrogates/$SURROGATE_CACHE"
    copy_dir_if_present "$src" "$RUN_ROOT/cache/surrogates/$SURROGATE_CACHE"
    src="$SOURCE_ROOT/cache/clean_victims/$VICTIM_CACHE"
    copy_dir_if_present "$src" "$RUN_ROOT/cache/clean_victims/$VICTIM_CACHE"

    # These pools were produced by the earlier cross-dataset runs. Failing here
    # is safer than letting every table-cell job independently retrain the same
    # pool (especially the ~47-minute Tiny ImageNet surrogates).
    s_count="$(cache_file_count "$RUN_ROOT/cache/surrogates/$SURROGATE_CACHE")"
    v_count="$(cache_file_count "$RUN_ROOT/cache/clean_victims/$VICTIM_CACHE")"
    [ "$s_count" -ge 20 ] || die "need 20 cached surrogates in cache/surrogates/$SURROGATE_CACHE; found $s_count"
    [ "$v_count" -ge "$NUM_VICTIMS" ] || die "need $NUM_VICTIMS cached clean victims in cache/clean_victims/$VICTIM_CACHE; found $v_count"
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
    local name
    name="$(run_name)"
    say "sync: extra-data cell -> $PERSIST_RESULT_ROOT/$name"
    if [ -d "$LOCAL_RESULT_ROOT/$name" ]; then
        mkdir -p "$PERSIST_RESULT_ROOT/$name"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$LOCAL_RESULT_ROOT/$name/" "$PERSIST_RESULT_ROOT/$name/"
    fi
    sync_cache_dir "$RUN_ROOT/cache/surrogates/$SURROGATE_CACHE" \
        "$SOURCE_ROOT/cache/surrogates/$SURROGATE_CACHE"
    sync_cache_dir "$RUN_ROOT/cache/clean_victims/$VICTIM_CACHE" \
        "$SOURCE_ROOT/cache/clean_victims/$VICTIM_CACHE"
}

handle_signal() {
    local signal="$1"
    say "signal: received $signal; stopping the job step before final sync"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

main() {
    local required name target_local status
    for required in DATASET MODEL CLASS_PAIR BUDGET ATTACK TARGET_FILE \
                    SURROGATE_CACHE VICTIM_CACHE VICTIM_LR; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done
    [ -n "${SLURM_TMPDIR:-}" ] || die 'SLURM_TMPDIR is unset; submit this file with sbatch'
    case "$ATTACK" in fc|gradmatch|sapa) ;; *) die "unsupported attack: $ATTACK" ;; esac
    case "$DATASET" in
        CIFAR100|TinyImageNet) NUM_TARGETS=4; NUM_VICTIMS=4 ;;
        SVHN) NUM_TARGETS=6; NUM_VICTIMS=5 ;;
        *) die "unsupported extra-data dataset: $DATASET" ;;
    esac

    module load python/3.11.5 cuda/12.6 cudnn
    [ -f "$ENV_ACTIVATE" ] || die "Python environment activation script missing: $ENV_ACTIVATE"
    source "$ENV_ACTIVATE"

    trap 'handle_signal USR1' USR1
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap sync_outputs EXIT

    stage_dataset
    stage_code_target_and_caches

    python - "$RUN_ROOT/target_sets/$(basename "$TARGET_FILE")" "$CLASS_PAIR" "$NUM_TARGETS" <<'PY'
import json
import sys

path, pair, required = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path) as handle:
    payload = json.load(handle)
indices = payload.get("pairs", {}).get(pair, {}).get("indices", [])
if len(indices) < required:
    raise SystemExit(
        "ERROR: pinned target file %s has %d target(s) for %s; need %d"
        % (path, len(indices), pair, required)
    )
print("pinned targets:", indices[:required])
PY

    name="$(run_name)"
    copy_dir_if_present "$PERSIST_RESULT_ROOT/$name" "$LOCAL_RESULT_ROOT/$name"
    target_local="$RUN_ROOT/target_sets/$(basename "$TARGET_FILE")"

    say "job: $SLURM_JOB_ID $SLURM_JOB_NAME on $(hostname)"
    say "cell: dataset=$DATASET model=$MODEL pair=$CLASS_PAIR budget=$BUDGET attack=$ATTACK selection=greedy jacobian=off"
    say "protocol: $NUM_TARGETS pinned targets x $NUM_VICTIMS victims; run=$name"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    local -a command
    command=(python "$RUN_ROOT/final_update.py"
        --dataset "$DATASET" --data_path "$LOCAL_DATA_ROOT" --seed 42
        --cache_dir "$RUN_ROOT/cache" --out_dir "$LOCAL_RESULT_ROOT"
        --model "$MODEL" --attack "$ATTACK" --base ours
        --base_dist cosine --lambda_margin 1.0
        --class_pair "$CLASS_PAIR" --pair_order poison-target
        --budget "$BUDGET" --epsilon 0.0313725
        --craft_steps 250 --craft_alpha 0.0039216 --restarts 8
        --craft_ensemble 5 --num_surrogates 20
        --surrogate_epochs 60 --surrogate_decay 35 45
        --target_select random --target_idx_file "$target_local"
        --num_targets "$NUM_TARGETS" --num_victims "$NUM_VICTIMS"
        --victim_epochs 50 --victim_lr "$VICTIM_LR" --victim_bs 125
        --victim_decay 40 --victim_wd 0.0 --clean_baseline --gpus all)
    if [ "$ATTACK" = sapa ]; then
        command+=(--sharp_mode worst --sharp_sigma 0.05)
    fi
    if [ "${CRAFT_LOWMEM:-0}" = 1 ] && [ "$ATTACK" != fc ]; then
        # craft_lowmem is the exact micro-batched objective; no fast-gradmatch
        # approximation is used in this table.
        command+=(--craft_lowmem --craft_batch "${CRAFT_BATCH:-256}")
    fi

    srun --ntasks=1 "${command[@]}" &
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
