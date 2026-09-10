#!/usr/bin/env bash
# Node-local runtime for one fully paired selection-architecture cell.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
RUN_ROOT="$SLURM_TMPDIR/PoisonBase"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
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
    rsync -a --ignore-existing --exclude='.lock' --exclude='*.tmp' "$src/" "$dst/"
}

stage_inputs() {
    local file selector_cache
    mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$RUN_ROOT/cache/surrogates" \
        "$RUN_ROOT/cache/clean_victims" "$RUN_ROOT/ours_result" \
        "$RUN_ROOT/target_sets"
    for file in final_update.py networks.py utils.py; do
        [ -f "$SOURCE_ROOT/$file" ] || die "required source file missing: $SOURCE_ROOT/$file"
        rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
    done
    [ -f "$SOURCE_ROOT/target_sets/ConvNetBN_gradmatch_dog-bird.json" ] || \
        die "missing pinned target file"
    rsync -a "$SOURCE_ROOT/target_sets/ConvNetBN_gradmatch_dog-bird.json" \
        "$RUN_ROOT/target_sets/"
    [ -d "$PERSIST_DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 input missing: $PERSIST_DATA_ROOT/cifar-10-batches-py"
    rsync -a "$PERSIST_DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"

    stage_dir_if_present \
        "$SOURCE_ROOT/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42" \
        "$RUN_ROOT/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42"
    if [ "$XFER_SELECTION" = basis ] && [ "$XFER_SELECTOR_MODEL" != ConvNetBN ]; then
        selector_cache="${XFER_SELECTOR_MODEL}_60ep_lr0.1_bs128_seed42"
        stage_dir_if_present "$SOURCE_ROOT/cache/surrogates/$selector_cache" \
                             "$RUN_ROOT/cache/surrogates/$selector_cache"
    fi
    stage_dir_if_present \
        "$SOURCE_ROOT/cache/clean_victims/ConvNetBN_50ep_lr0.1_bs125_wd0_seed42" \
        "$RUN_ROOT/cache/clean_victims/ConvNetBN_50ep_lr0.1_bs125_wd0_seed42"
    stage_dir_if_present "$SOURCE_ROOT/ours_result/$XFER_RUN_NAME" \
                         "$RUN_ROOT/ours_result/$XFER_RUN_NAME"
}

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    say "sync: paired architecture-transfer output -> $SOURCE_ROOT"
    if [ -d "$RUN_ROOT/ours_result/$XFER_RUN_NAME" ]; then
        mkdir -p "$SOURCE_ROOT/ours_result/$XFER_RUN_NAME"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$RUN_ROOT/ours_result/$XFER_RUN_NAME/" \
            "$SOURCE_ROOT/ours_result/$XFER_RUN_NAME/"
    fi
    sync_cache_dir \
        "$RUN_ROOT/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42" \
        "$SOURCE_ROOT/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42"
    if [ "$XFER_SELECTION" = basis ] && [ "$XFER_SELECTOR_MODEL" != ConvNetBN ]; then
        local selector_cache="${XFER_SELECTOR_MODEL}_60ep_lr0.1_bs128_seed42"
        sync_cache_dir "$RUN_ROOT/cache/surrogates/$selector_cache" \
                       "$SOURCE_ROOT/cache/surrogates/$selector_cache"
    fi
    sync_cache_dir \
        "$RUN_ROOT/cache/clean_victims/ConvNetBN_50ep_lr0.1_bs125_wd0_seed42" \
        "$SOURCE_ROOT/cache/clean_victims/ConvNetBN_50ep_lr0.1_bs125_wd0_seed42"
    say "sync: complete"
}

handle_signal() {
    say "signal: received $1; stopping job step before final sync"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

main() {
    local required base selector_args=() sharp_args=()
    [ -n "${SLURM_TMPDIR:-}" ] || die "SLURM_TMPDIR is unset; use sbatch"
    for required in XFER_ATTACK XFER_BUDGET XFER_SELECTION XFER_SELECTOR_MODEL \
                    XFER_RUN_NAME ORIGINAL_COMMAND; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done
    case "$XFER_ATTACK" in gradmatch|sapa) ;; *) die "bad attack: $XFER_ATTACK" ;; esac
    case "$XFER_SELECTION" in random|basis) ;; *) die "bad selection: $XFER_SELECTION" ;; esac
    case "$XFER_SELECTOR_MODEL" in shared|ConvNetBN|ResNet20BN|VGG13BN) ;;
        *) die "bad selector architecture: $XFER_SELECTOR_MODEL" ;;
    esac
    if [ "$XFER_SELECTION" = random ] && [ "$XFER_SELECTOR_MODEL" != shared ]; then
        die "RAND must use XFER_SELECTOR_MODEL=shared"
    fi
    if [ "$XFER_SELECTION" = basis ] && [ "$XFER_SELECTOR_MODEL" = shared ]; then
        die "BASIS requires a concrete selector architecture"
    fi

    if command -v module >/dev/null 2>&1; then
        module load python/3.11.5 cuda/12.6 cudnn
    fi
    [ -f "$ENV_ACTIVATE" ] || die "environment activation missing: $ENV_ACTIVATE"
    source "$ENV_ACTIVATE"
    trap 'handle_signal USR1' USR1
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap sync_outputs EXIT
    stage_inputs

    base=random
    if [ "$XFER_SELECTION" = basis ]; then
        base=ours
        selector_args=(--sel_model "$XFER_SELECTOR_MODEL" --base_dist cosine --lambda_margin 1)
    fi
    if [ "$XFER_ATTACK" = sapa ]; then
        sharp_args=(--sharp_mode worst --sharp_sigma 0.05)
    fi

    say "job: $SLURM_JOB_ID $SLURM_JOB_NAME on $(hostname)"
    say "config: $ORIGINAL_COMMAND"
    say "pairing: A=V=ConvNetBN, S=$XFER_SELECTOR_MODEL, K=20, 6 targets x 6 victims"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    srun --ntasks=1 python "$RUN_ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$LOCAL_DATA_ROOT" --seed 42 \
        --cache_dir "$RUN_ROOT/cache" --out_dir "$RUN_ROOT/ours_result" \
        --model ConvNetBN --attack "$XFER_ATTACK" --base "$base" \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$XFER_BUDGET" --epsilon 0.0313725 \
        --craft_steps 750 --craft_alpha 0.0039216 --restarts 8 \
        --craft_ensemble 5 --sel_K 20 "${selector_args[@]}" "${sharp_args[@]}" \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 6 --target_select 70 \
        --target_idx_file "$RUN_ROOT/target_sets/ConvNetBN_gradmatch_dog-bird.json" \
        --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0 --clean_baseline &
    STEP_PID=$!
    set +e
    wait "$STEP_PID"
    local status=$?
    set -e
    STEP_PID=""
    sync_outputs
    trap - EXIT
    exit "$status"
}

main "$@"
