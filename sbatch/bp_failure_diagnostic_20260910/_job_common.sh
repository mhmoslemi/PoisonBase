#!/usr/bin/env bash
# Node-local runtime for one ResNet20/BP failure-diagnostic configuration.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
RUN_ROOT="$SLURM_TMPDIR/PoisonBase"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
OUTPUT_NAME="ResNet20BN_fc_${DIAG_SELECTION}_dog-bird_b${DIAG_BUDGET}"
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

stage_inputs() {
    local file
    mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$RUN_ROOT/cache/surrogates" \
        "$RUN_ROOT/ours_result" "$RUN_ROOT/target_sets" \
        "$RUN_ROOT/bp_failure_diagnostic"
    for file in final_update.py bp_failure_diagnostic.py networks.py utils.py; do
        [ -f "$SOURCE_ROOT/$file" ] || die "required source file missing: $SOURCE_ROOT/$file"
        rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
    done
    [ -f "$SOURCE_ROOT/target_sets/ResNet20BN_fc_dog-bird.json" ] || \
        die "missing pinned BP target file"
    rsync -a "$SOURCE_ROOT/target_sets/ResNet20BN_fc_dog-bird.json" \
        "$RUN_ROOT/target_sets/"
    [ -d "$PERSIST_DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 input missing: $PERSIST_DATA_ROOT/cifar-10-batches-py"
    rsync -a "$PERSIST_DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"
    stage_dir_if_present \
        "$SOURCE_ROOT/cache/surrogates/ResNet20BN_60ep_lr0.1_bs128_seed42" \
        "$RUN_ROOT/cache/surrogates/ResNet20BN_60ep_lr0.1_bs128_seed42"
    [ -d "$SOURCE_ROOT/ours_result/$DIAG_ASR_RUN_NAME" ] || \
        die "completed-ASR input missing: $SOURCE_ROOT/ours_result/$DIAG_ASR_RUN_NAME"
    stage_dir_if_present "$SOURCE_ROOT/ours_result/$DIAG_ASR_RUN_NAME" \
                         "$RUN_ROOT/ours_result/$DIAG_ASR_RUN_NAME"
    stage_dir_if_present "$SOURCE_ROOT/bp_failure_diagnostic/$OUTPUT_NAME" \
                         "$RUN_ROOT/bp_failure_diagnostic/$OUTPUT_NAME"
}

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    say "sync: BP diagnostic -> $SOURCE_ROOT/bp_failure_diagnostic/$OUTPUT_NAME"
    if [ -d "$RUN_ROOT/bp_failure_diagnostic/$OUTPUT_NAME" ]; then
        mkdir -p "$SOURCE_ROOT/bp_failure_diagnostic/$OUTPUT_NAME"
        rsync -a --exclude='*.tmp' \
            "$RUN_ROOT/bp_failure_diagnostic/$OUTPUT_NAME/" \
            "$SOURCE_ROOT/bp_failure_diagnostic/$OUTPUT_NAME/"
    fi
    say "sync: complete"
}

handle_signal() {
    say "signal: received $1; stopping diagnostic before final sync"
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
    [ -n "${SLURM_TMPDIR:-}" ] || die "SLURM_TMPDIR is unset; use sbatch"
    for required in DIAG_SELECTION DIAG_BUDGET DIAG_ASR_RUN_NAME ORIGINAL_COMMAND; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done
    case "$DIAG_SELECTION" in random|basis) ;; *) die "bad selector: $DIAG_SELECTION" ;; esac
    case "$DIAG_BUDGET" in 0.005|0.02) ;; *) die "bad diagnostic budget: $DIAG_BUDGET" ;; esac

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

    say "job: $SLURM_JOB_ID $SLURM_JOB_NAME on $(hostname)"
    say "config: $ORIGINAL_COMMAND"
    say "diagnostic only: 10 targets, poison construction rerun, zero victim training"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    srun --ntasks=1 python "$RUN_ROOT/bp_failure_diagnostic.py" \
        --selection "$DIAG_SELECTION" --budget "$DIAG_BUDGET" \
        --data-path "$LOCAL_DATA_ROOT" --cache-dir "$RUN_ROOT/cache" \
        --target-file "$RUN_ROOT/target_sets/ResNet20BN_fc_dog-bird.json" \
        --asr-run-dir "$RUN_ROOT/ours_result/$DIAG_ASR_RUN_NAME" \
        --output-dir "$RUN_ROOT/bp_failure_diagnostic/$OUTPUT_NAME" \
        --num-surrogates 20 --craft-ensemble 5 --craft-steps 250 \
        --craft-alpha 0.0039216 --fc-restarts 1 --curve-every 25 &
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
