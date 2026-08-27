#!/usr/bin/env bash
# Shared runtime for the expanded cross-architecture jobs. Each generated
# sbatch file exports exactly one table-cell configuration and sources this.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-/home/mmoslem3/scratch/data}"
PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
RUN_ROOT="$SLURM_TMPDIR/attack_if"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
H200_ROOT="$SOURCE_ROOT/last_night_H200_2026-08-26"
SYNCED=0
STEP_PID=""

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

stage_dir_if_present() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        rsync -a --exclude='.lock' --exclude='*.tmp' "$src/" "$dst/"
    else
        say "stage: optional directory absent: $src"
    fi
}

sync_cache_dir() {
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    rsync -a --ignore-existing --exclude='.lock' --exclude='*.tmp' \
        "$src/" "$dst/"
}

stage_inputs() {
    local file target_attack target_file
    mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$RUN_ROOT/cache/surrogates" \
        "$RUN_ROOT/cache/clean_victims" "$RUN_ROOT/ours_result" \
        "$RUN_ROOT/target_sets"

    # Only the four source files used by this launcher enter node-local storage.
    for file in final_update.py networks.py utils.py cross_arch.sh; do
        [ -f "$SOURCE_ROOT/$file" ] || \
            die "required source file missing: $SOURCE_ROOT/$file"
        rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
    done

    [ -d "$PERSIST_DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 input missing: $PERSIST_DATA_ROOT/cifar-10-batches-py"
    rsync -a "$PERSIST_DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"

    # SAPA shares GM's pinned targets; both budgets use the original b0.005 set.
    target_attack="$CROSS_ATTACK"
    [ "$target_attack" = sapa ] && target_attack=gradmatch
    target_file="xarch_${CROSS_MODEL}_${target_attack}_dog-bird_b0.005.json"
    [ -s "$SOURCE_ROOT/target_sets/$target_file" ] || \
        die "pinned target file missing: $SOURCE_ROOT/target_sets/$target_file"
    rsync -a "$SOURCE_ROOT/target_sets/$target_file" "$RUN_ROOT/target_sets/"

    stage_dir_if_present \
        "$SOURCE_ROOT/cache/surrogates/${CROSS_MODEL}_60ep_lr0.1_bs128_seed42" \
        "$RUN_ROOT/cache/surrogates/${CROSS_MODEL}_60ep_lr0.1_bs128_seed42"
    if [ "$CROSS_SELECTION" != random ] && \
       [ "$CROSS_SELECTOR_MODEL" != "$CROSS_MODEL" ]; then
        stage_dir_if_present \
            "$SOURCE_ROOT/cache/surrogates/${CROSS_SELECTOR_MODEL}_60ep_lr0.1_bs128_seed42" \
            "$RUN_ROOT/cache/surrogates/${CROSS_SELECTOR_MODEL}_60ep_lr0.1_bs128_seed42"
    fi
    stage_dir_if_present \
        "$SOURCE_ROOT/cache/clean_victims/${CROSS_MODEL}_50ep_lr0.1_bs125_wd0_seed42" \
        "$RUN_ROOT/cache/clean_victims/${CROSS_MODEL}_50ep_lr0.1_bs125_wd0_seed42"

    # H200 shards are staged first, then any newer copy in the main result tree.
    # final_update.py resumes completed target/victim trials and poison caches.
    stage_dir_if_present "$H200_ROOT/ours_result/$CROSS_RUN_NAME" \
                         "$RUN_ROOT/ours_result/$CROSS_RUN_NAME"
    stage_dir_if_present "$SOURCE_ROOT/ours_result/$CROSS_RUN_NAME" \
                         "$RUN_ROOT/ours_result/$CROSS_RUN_NAME"
}

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    say "sync: cross-architecture artifacts -> $SOURCE_ROOT"

    if [ -d "$RUN_ROOT/ours_result/$CROSS_RUN_NAME" ]; then
        mkdir -p "$SOURCE_ROOT/ours_result/$CROSS_RUN_NAME"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$RUN_ROOT/ours_result/$CROSS_RUN_NAME/" \
            "$SOURCE_ROOT/ours_result/$CROSS_RUN_NAME/"
    fi
    sync_cache_dir \
        "$RUN_ROOT/cache/surrogates/${CROSS_MODEL}_60ep_lr0.1_bs128_seed42" \
        "$SOURCE_ROOT/cache/surrogates/${CROSS_MODEL}_60ep_lr0.1_bs128_seed42"
    if [ "$CROSS_SELECTION" != random ] && \
       [ "$CROSS_SELECTOR_MODEL" != "$CROSS_MODEL" ]; then
        sync_cache_dir \
            "$RUN_ROOT/cache/surrogates/${CROSS_SELECTOR_MODEL}_60ep_lr0.1_bs128_seed42" \
            "$SOURCE_ROOT/cache/surrogates/${CROSS_SELECTOR_MODEL}_60ep_lr0.1_bs128_seed42"
    fi
    sync_cache_dir \
        "$RUN_ROOT/cache/clean_victims/${CROSS_MODEL}_50ep_lr0.1_bs125_wd0_seed42" \
        "$SOURCE_ROOT/cache/clean_victims/${CROSS_MODEL}_50ep_lr0.1_bs125_wd0_seed42"
    say "sync: complete"
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
    local required
    [ -n "${SLURM_TMPDIR:-}" ] || \
        die "SLURM_TMPDIR is unset; submit this file with sbatch"
    for required in CROSS_MODEL CROSS_SELECTOR_MODEL CROSS_ATTACK CROSS_SELECTION \
                    CROSS_BUDGET CROSS_K CROSS_NUM_TARGETS CROSS_NUM_VICTIMS \
                    CROSS_RUN_NAME ORIGINAL_COMMAND; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done

    if command -v module >/dev/null 2>&1; then
        module load python/3.11.5 cuda/12.6 cudnn
    else
        say "environment modules unavailable; using $PYTHON_ENV directly"
    fi
    source "$PYTHON_ENV/bin/activate"

    trap 'handle_signal USR1' USR1
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap sync_outputs EXIT

    stage_inputs

    say "job: $SLURM_JOB_ID $SLURM_JOB_NAME on $(hostname)"
    say "work: $RUN_ROOT"
    say "config: $ORIGINAL_COMMAND"
    say "protocol: one cell, ${CROSS_NUM_TARGETS} targets x ${CROSS_NUM_VICTIMS} victims"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    srun --ntasks=1 env \
        OSTYPE="${OSTYPE:-linux-gnu}" \
        PROJECT_DIR="$RUN_ROOT" \
        DATA_PATH="$LOCAL_DATA_ROOT" \
        CACHE_DIR="$RUN_ROOT/cache" \
        OUT_DIR="$RUN_ROOT/ours_result" \
        VENV_ACTIVATE="" \
        MODELS="$CROSS_MODEL" \
        SELECTOR_MODELS="$CROSS_SELECTOR_MODEL" \
        ATTACKS="$CROSS_ATTACK" \
        SELECTIONS="$CROSS_SELECTION" \
        BUDGET="$CROSS_BUDGET" \
        TARGET_SET_BUDGET=0.005 \
        SEL_K="$CROSS_K" \
        RUN_MATCHED=1 \
        NUM_TARGETS="$CROSS_NUM_TARGETS" \
        NUM_VICTIMS="$CROSS_NUM_VICTIMS" \
        sh "$RUN_ROOT/cross_arch.sh" &
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
