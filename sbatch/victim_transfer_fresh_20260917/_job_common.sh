#!/usr/bin/env bash
# Shared runtime for one fresh S=A -> V architecture-transfer configuration.
#
# K changes only the selector ensemble. Poison construction always uses the
# first five checkpoints of A, and victim training uses V. Normal submissions
# are fresh. Recovery submissions set VXF_RESUME=1, stage only that run's saved
# state, and omit --FORCE so completed target/victim pairs are preserved.

set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/victim_transfer_fresh_20260917_result}"
RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/PoisonBase_victim_transfer_fresh}"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
LOCAL_CACHE_ROOT="$RUN_ROOT/cache"
LOCAL_RESULT_ROOT="$RUN_ROOT/victim_transfer_fresh_20260917_result"
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

cache_has_nets() {
    local dir="$1" count="$2" i
    for ((i = 0; i < count; i++)); do
        [ -s "$dir/net_${i}.pt" ] || return 1
    done
}

precompute_cache() {
    local model="$1" part="$2"
    say "cache: creating missing $part checkpoints for $model"
    run_tracked python "$ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$DATA_ROOT" --seed 42 --gpus all \
        --cache_dir "$CACHE_ROOT" --out_dir "$RESULT_ROOT" \
        --model "$model" --victim_model "$model" \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
        --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
        --num_victims 6 --victim_epochs 50 --victim_lr 0.1 \
        --victim_bs 125 --victim_decay 40 --victim_wd 0 \
        --precompute_only --precompute_part "$part"
}

ensure_cache_part() {
    local model="$1" part="$2" dir count lock_file lock_fd
    if [ "$part" = surrogate ]; then
        dir="$CACHE_ROOT/surrogates/${model}_60ep_lr0.1_bs128_seed42"
        count=20
    else
        dir="$CACHE_ROOT/clean_victims/${model}_50ep_lr0.1_bs125_wd0_seed42"
        count=6
    fi
    mkdir -p "$CACHE_ROOT/.victim_transfer_fresh_locks"
    lock_file="$CACHE_ROOT/.victim_transfer_fresh_locks/${model}_${part}.lock"
    exec {lock_fd}>"$lock_file"
    flock -x "$lock_fd"
    if ! cache_has_nets "$dir" "$count"; then
        precompute_cache "$model" "$part" || \
            die "$part checkpoint creation failed for $model"
    fi
    cache_has_nets "$dir" "$count" || \
        die "$part cache for $model is incomplete after precompute"
    flock -u "$lock_fd"
    exec {lock_fd}>&-
}

stage_dir_if_present() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        rsync -a --exclude='.lock' --exclude='*.tmp' "$src/" "$dst/"
    fi
}

target_attack() {
    # SAPA and GM use the same pinned targets in the existing protocol.
    printf '%s\n' gradmatch
}

stage_inputs() {
    local file target_file cache_name
    mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$LOCAL_CACHE_ROOT/surrogates" \
        "$LOCAL_CACHE_ROOT/clean_victims" "$LOCAL_RESULT_ROOT" \
        "$RUN_ROOT/target_sets"

    for file in final_update.py networks.py utils.py; do
        [ -f "$ROOT/$file" ] || die "required source file missing: $ROOT/$file"
        rsync -a "$ROOT/$file" "$RUN_ROOT/"
    done

    target_file="${VXF_SOURCE_MODEL}_$(target_attack)_dog-bird.json"
    [ -s "$ROOT/target_sets/$target_file" ] || \
        die "pinned target set missing: $ROOT/target_sets/$target_file"
    rsync -a "$ROOT/target_sets/$target_file" "$RUN_ROOT/target_sets/"

    [ -d "$DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 data missing: $DATA_ROOT/cifar-10-batches-py"
    rsync -a "$DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"

    cache_name="${VXF_SOURCE_MODEL}_60ep_lr0.1_bs128_seed42"
    stage_dir_if_present "$CACHE_ROOT/surrogates/$cache_name" \
                         "$LOCAL_CACHE_ROOT/surrogates/$cache_name"
    cache_name="${VXF_VICTIM_MODEL}_50ep_lr0.1_bs125_wd0_seed42"
    stage_dir_if_present "$CACHE_ROOT/clean_victims/$cache_name" \
                         "$LOCAL_CACHE_ROOT/clean_victims/$cache_name"

    if [ "${VXF_RESUME:-0}" = 1 ]; then
        # Only the matching run is staged. This preserves its completed rows and
        # poison cache without importing state from another configuration.
        stage_dir_if_present "$RESULT_ROOT/$VXF_RUN_NAME" \
                             "$LOCAL_RESULT_ROOT/$VXF_RUN_NAME"
    fi
}

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    if [ -d "$LOCAL_RESULT_ROOT/$VXF_RUN_NAME" ]; then
        say "sync: $VXF_RUN_NAME -> $RESULT_ROOT"
        mkdir -p "$RESULT_ROOT/$VXF_RUN_NAME"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$LOCAL_RESULT_ROOT/$VXF_RUN_NAME/" \
            "$RESULT_ROOT/$VXF_RUN_NAME/"
    fi
}

handle_signal() {
    say "signal: received $1; stopping active step and syncing partial output"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
        STEP_PID=""
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

verify_target_file() {
    local path="$1"
    python - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as handle:
    payload = json.load(handle)
indices = payload['pairs']['dog-bird']['indices']
if len(indices) != 10 or len(set(map(int, indices))) != 10:
    raise SystemExit('%s must contain exactly 10 unique dog-bird targets' % path)
print('targets: 10 pinned IDs:', ' '.join(map(str, indices)))
PY
}

verify_results() {
    local run_dir="$LOCAL_RESULT_ROOT/$VXF_RUN_NAME"
    local target_file="$RUN_ROOT/target_sets/${VXF_SOURCE_MODEL}_$(target_attack)_dog-bird.json"
    python - "$run_dir" "$target_file" "$VXF_VICTIM_MODEL" <<'PY'
import csv
import json
import os
import sys
from collections import Counter

run_dir, target_path, victim_model = sys.argv[1:]
with open(target_path) as handle:
    expected = list(map(int, json.load(handle)['pairs']['dog-bird']['indices']))
with open(os.path.join(run_dir, 'results.csv'), newline='') as handle:
    rows = list(csv.DictReader(handle))

pairs = []
for row in rows:
    target = int(row['target_idx'])
    victim = int(row['victim_id'])
    success = int(row['success'])
    if success not in (0, 1):
        raise SystemExit('success must be binary')
    if row['model'] != victim_model:
        raise SystemExit('result model %s != requested victim %s' %
                         (row['model'], victim_model))
    pairs.append((target, victim))

problems = []
if len(rows) != 60:
    problems.append('expected 60 rows, found %d' % len(rows))
if len(set(pairs)) != 60:
    problems.append('expected 60 unique target/victim pairs, found %d' % len(set(pairs)))
if {target for target, _ in pairs} != set(expected):
    problems.append('target IDs differ from pinned source-architecture set')
for target in expected:
    got = sorted(v for t, v in pairs if t == target)
    if got != list(range(6)):
        problems.append('target %d has victim IDs %s' % (target, got))

cache_dir = os.path.join(run_dir, 'poison_cache')
base_count = sum(os.path.isfile(os.path.join(cache_dir, 'base_%d.json' % t))
                 for t in expected)
delta_count = sum(os.path.isfile(os.path.join(cache_dir, 'delta_%d.pt' % t))
                  for t in expected)
if base_count != 10 or delta_count != 10:
    problems.append('expected 10 fresh base files and 10 fresh delta files; got %d/%d' %
                    (base_count, delta_count))
if problems:
    raise SystemExit('incomplete result: ' + '; '.join(problems))
print('verified: 10 targets x 6 victims = 60 evaluations; 10 poison caches')
PY
}

main() {
    local required expected_degree target_file status
    local memory_args=() sharp_args=() force_args=()

    [ -n "${SLURM_TMPDIR:-}" ] || die "SLURM_TMPDIR is unset; submit with sbatch"
    for required in VXF_INDEX VXF_ATTACK VXF_BUDGET VXF_K \
                    VXF_SOURCE_MODEL VXF_VICTIM_MODEL VXF_TARGET_DEGREE \
                    VXF_RUN_NAME ORIGINAL_COMMAND; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done
    case "$VXF_ATTACK" in gradmatch|sapa) ;; *) die "bad attack: $VXF_ATTACK" ;; esac
    case "$VXF_BUDGET" in 0.002|0.005) ;; *) die "bad budget: $VXF_BUDGET" ;; esac
    case "$VXF_K" in 10|20) ;; *) die "bad selector K: $VXF_K" ;; esac
    case "$VXF_SOURCE_MODEL" in ConvNetBN) expected_degree=70 ;; ResNet20BN) expected_degree=14 ;; VGG13BN) expected_degree=50 ;; *) die "bad source model: $VXF_SOURCE_MODEL" ;; esac
    case "$VXF_VICTIM_MODEL" in ConvNetBN|ResNet20BN|VGG13BN) ;; *) die "bad victim model: $VXF_VICTIM_MODEL" ;; esac
    case "${VXF_RESUME:-0}" in 0) force_args=(--FORCE) ;; 1) ;; *) die "VXF_RESUME must be 0 or 1" ;; esac
    [ "$VXF_TARGET_DEGREE" = "$expected_degree" ] || \
        die "target degree $VXF_TARGET_DEGREE != expected $expected_degree"

    if command -v module >/dev/null 2>&1; then
        module load python/3.11.5 cuda/12.6 cudnn
    fi
    command -v flock >/dev/null 2>&1 || die "flock is required"
    [ -f "$ENV_ACTIVATE" ] || die "environment activation missing: $ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    trap 'handle_signal USR1' USR1
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap sync_outputs EXIT

    [ -f "$ROOT/final_update.py" ] || die "missing $ROOT/final_update.py"
    ensure_cache_part "$VXF_SOURCE_MODEL" surrogate
    ensure_cache_part "$VXF_VICTIM_MODEL" victim
    stage_inputs

    target_file="$RUN_ROOT/target_sets/${VXF_SOURCE_MODEL}_$(target_attack)_dog-bird.json"
    verify_target_file "$target_file"

    if [ "$VXF_SOURCE_MODEL" = VGG13BN ]; then
        memory_args=(--craft_lowmem --craft_batch 256 --fast_gradmatch)
    fi
    if [ "$VXF_ATTACK" = sapa ]; then
        sharp_args=(--sharp_mode worst --sharp_sigma 0.05)
    fi

    say "job: ${SLURM_JOB_ID:-unknown} ${SLURM_JOB_NAME:-unknown} on $(hostname)"
    say "config: $ORIGINAL_COMMAND"
    say "protocol: S=A=$VXF_SOURCE_MODEL V=$VXF_VICTIM_MODEL K=$VXF_K"
    if [ "${VXF_RESUME:-0}" = 1 ]; then
        say "protocol: resume this configuration from saved rows/cache"
    else
        say "protocol: fresh base selection + fresh poison optimization (FORCE)"
    fi
    say "output: $RESULT_ROOT/$VXF_RUN_NAME"

    set +e
    run_tracked srun --ntasks=1 python "$RUN_ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$LOCAL_DATA_ROOT" --seed 42 --gpus all \
        --cache_dir "$LOCAL_CACHE_ROOT" --out_dir "$LOCAL_RESULT_ROOT" \
        --model "$VXF_SOURCE_MODEL" --victim_model "$VXF_VICTIM_MODEL" \
        --attack "$VXF_ATTACK" --base ours --base_dist cosine --lambda_margin 1 \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$VXF_BUDGET" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 --restarts 8 --fc_restarts 1 \
        --craft_ensemble 5 --craft_aug "${memory_args[@]}" "${sharp_args[@]}" \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
        --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
        --sel_K "$VXF_K" \
        --num_targets 10 --target_select "$VXF_TARGET_DEGREE" \
        --target_idx_file "$target_file" --keep_pinned_targets \
        --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0 --clean_baseline "${force_args[@]}"
    status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        verify_results
    fi
    sync_outputs
    trap - EXIT
    exit "$status"
}

main "$@"
