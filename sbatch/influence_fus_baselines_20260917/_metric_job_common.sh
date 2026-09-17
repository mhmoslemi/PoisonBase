#!/usr/bin/env bash
# Train one of the 20 clean surrogate trajectories used by the Gao selectors.
set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/PoisonBase_ifb_metric}"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
LOCAL_CACHE_ROOT="$RUN_ROOT/cache"
METRIC_TAG="CIFAR10_${METRIC_MODEL}_60ep_select10_lr0.1_bs128_decay35-45_wd0_seed42/class5"
STEP_PID=""
SYNCED=0

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

run_tracked() {
    "$@" &
    STEP_PID=$!
    local status=0
    wait "$STEP_PID" || status=$?
    STEP_PID=""
    return "$status"
}

sync_output() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    local src="$LOCAL_CACHE_ROOT/selector_metrics/$METRIC_TAG"
    local dst="$CACHE_ROOT/selector_metrics/$METRIC_TAG"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        rsync -a --exclude='*.tmp*' "$src/" "$dst/"
    fi
}

handle_signal() {
    say "signal: received $1"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
        STEP_PID=""
    fi
    sync_output
    trap - EXIT
    exit 143
}

main() {
    local file persistent_dir local_dir shard resource_shard status time_file
    local time_prefix=()
    [ -n "${SLURM_TMPDIR:-}" ] || die "SLURM_TMPDIR is unset; submit with sbatch"
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || die "SLURM_ARRAY_TASK_ID is unset"
    case "$METRIC_MODEL" in ConvNetBN|ResNet20BN) ;; *) die "bad model: $METRIC_MODEL" ;; esac
    case "$SLURM_ARRAY_TASK_ID" in [0-9]|1[0-9]) ;; *) die "array ID must be 0..19" ;; esac

    if command -v module >/dev/null 2>&1; then
        module load python/3.11.5 cuda/12.6 cudnn
    fi
    [ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    trap 'handle_signal USR1' USR1
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap sync_output EXIT

    mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$LOCAL_CACHE_ROOT/selector_metrics"
    for file in final_update.py precompute_selector_metrics.py networks.py utils.py; do
        [ -f "$ROOT/$file" ] || die "missing $ROOT/$file"
        rsync -a "$ROOT/$file" "$RUN_ROOT/"
    done
    [ -d "$DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 data missing: $DATA_ROOT/cifar-10-batches-py"
    rsync -a "$DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"

    persistent_dir="$CACHE_ROOT/selector_metrics/$METRIC_TAG"
    local_dir="$LOCAL_CACHE_ROOT/selector_metrics/$METRIC_TAG"
    shard="net_${SLURM_ARRAY_TASK_ID}.npz"
    resource_shard="net_${SLURM_ARRAY_TASK_ID}.resources.json"
    mkdir -p "$local_dir"
    if [ -s "$persistent_dir/$shard" ]; then
        rsync -a "$persistent_dir/$shard" "$local_dir/"
    fi
    if [ -s "$persistent_dir/$resource_shard" ]; then
        rsync -a "$persistent_dir/$resource_shard" "$local_dir/"
    fi

    time_file="$local_dir/net_${SLURM_ARRAY_TASK_ID}.gnu_time_${SLURM_JOB_ID:-manual}.txt"
    if [ -x /usr/bin/time ]; then
        time_prefix=(/usr/bin/time -v -o "$time_file")
        say "whole-shard CPU/time resources: $time_file"
    else
        say "warning: /usr/bin/time is unavailable; Python resource metrics remain enabled"
    fi

    say "metric: model=$METRIC_MODEL shard=$SLURM_ARRAY_TASK_ID/19"
    set +e
    run_tracked srun --ntasks=1 "${time_prefix[@]}" \
        python "$RUN_ROOT/precompute_selector_metrics.py" \
        --dataset CIFAR10 --data-path "$LOCAL_DATA_ROOT" \
        --cache-dir "$LOCAL_CACHE_ROOT" \
        --selector-metric-dir "$LOCAL_CACHE_ROOT/selector_metrics" \
        --model "$METRIC_MODEL" --class-pair dog-bird \
        --surrogate-id "$SLURM_ARRAY_TASK_ID" --seed 42 \
        --epochs 60 --select-epoch 10 --lr 0.1 --batch-size 128 \
        --grad-batch-size 8 --decay 35 45 --weight-decay 0
    status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        [ -s "$local_dir/$shard" ] || die "metric program returned success without $shard"
        [ -s "$local_dir/$resource_shard" ] || \
            die "metric program returned success without $resource_shard"
    fi
    sync_output
    trap - EXIT
    exit "$status"
}

main "$@"
