#!/usr/bin/env bash
# Shared node-local runtime for one BASIS cross-architecture/K configuration.
#
# Invariants for this experiment:
#   * pinned targets and six victim seeds (10 targets by default; reruns may use 8)
#   * A=V for poison crafting and victim training
#   * S affects base selection only
#   * K affects the number of selector checkpoints only
#   * the first five V checkpoints craft poisons for every K
#   * method=ours and Jacobian disabled
#   * score/victim-training settings default to the original table protocol,
#     but may be pinned explicitly by a dedicated rerun batch
#   * 250 poison-optimization steps

set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/cross_arch_k_10x6_result}"
RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/PoisonBase_cross_arch_k_10x6}"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
LOCAL_CACHE_ROOT="$RUN_ROOT/cache"
LOCAL_RESULT_ROOT="$RUN_ROOT/cross_arch_k_10x6_result"
XFULL_NUM_TARGETS="${XFULL_NUM_TARGETS:-10}"
XFULL_NUM_VICTIMS="${XFULL_NUM_VICTIMS:-6}"
XFULL_COMPONENT="${XFULL_COMPONENT:-}"
XFULL_BASE_DIST="${XFULL_BASE_DIST:-cosine}"
XFULL_LAMBDA_MARGIN="${XFULL_LAMBDA_MARGIN:-1}"
XFULL_VICTIM_EPOCHS="${XFULL_VICTIM_EPOCHS:-50}"
XFULL_VICTIM_DECAY="${XFULL_VICTIM_DECAY:-40}"
SYNCED=0
STEP_PID=""

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

run_tracked() {
    "$@" &
    STEP_PID=$!
    set +e
    wait "$STEP_PID"
    local status=$?
    set -e
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
    say "cache: completing $part checkpoints for $model"
    run_tracked python "$ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$DATA_ROOT" --seed 42 \
        --cache_dir "$CACHE_ROOT" --out_dir "$RESULT_ROOT" \
        --model "$model" --gpus all \
        --num_surrogates 30 --surrogate_epochs 60 --surrogate_lr 0.1 \
        --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
        --num_victims 6 --victim_epochs "$XFULL_VICTIM_EPOCHS" --victim_lr 0.1 \
        --victim_bs 125 --victim_decay "$XFULL_VICTIM_DECAY" --victim_wd 0 \
        --precompute_only --precompute_part "$part"
}

# Serialize completion of persistent model caches. Without this, many K=30 jobs
# launched together could independently train net_20..net_29 on node-local disks.
ensure_model_cache() {
    local model="$1" need_victims="$2"
    local surrogate_dir victim_dir lock_file lock_fd
    surrogate_dir="$CACHE_ROOT/surrogates/${model}_60ep_lr0.1_bs128_seed42"
    victim_dir="$CACHE_ROOT/clean_victims/${model}_${XFULL_VICTIM_EPOCHS}ep_lr0.1_bs125_wd0_seed42"
    mkdir -p "$CACHE_ROOT/.cross_arch_k_locks"
    lock_file="$CACHE_ROOT/.cross_arch_k_locks/${model}.lock"
    exec {lock_fd}>"$lock_file"
    say "cache: waiting for shared $model checkpoint lock"
    flock -x "$lock_fd"

    if ! cache_has_nets "$surrogate_dir" 30; then
        precompute_cache "$model" surrogate || die "surrogate precompute failed for $model"
    fi
    cache_has_nets "$surrogate_dir" 30 || \
        die "$model surrogate cache is incomplete after precompute"

    if [ "$need_victims" = 1 ] && ! cache_has_nets "$victim_dir" 6; then
        precompute_cache "$model" victim || die "clean-victim precompute failed for $model"
    fi
    if [ "$need_victims" = 1 ]; then
        cache_has_nets "$victim_dir" 6 || \
            die "$model clean-victim cache is incomplete after precompute"
    fi

    flock -u "$lock_fd"
    exec {lock_fd}>&-
    say "cache: $model checkpoints ready"
}

stage_dir_if_present() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        rsync -a --exclude='.lock' --exclude='*.tmp' "$src/" "$dst/"
    fi
}

sync_cache_dir() {
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    rsync -a --ignore-existing --exclude='.lock' --exclude='*.tmp' "$src/" "$dst/"
}

target_attack() {
    if [ "$XFULL_ATTACK" = sapa ]; then
        printf '%s\n' gradmatch
    else
        printf '%s\n' "$XFULL_ATTACK"
    fi
}

stage_inputs() {
    local file lookup target_file model cache_name
    mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$LOCAL_CACHE_ROOT/surrogates" \
        "$LOCAL_CACHE_ROOT/clean_victims" "$LOCAL_RESULT_ROOT" "$RUN_ROOT/target_sets"

    for file in final_update.py networks.py utils.py; do
        [ -f "$ROOT/$file" ] || die "required source file missing: $ROOT/$file"
        rsync -a "$ROOT/$file" "$RUN_ROOT/"
    done

    lookup="$(target_attack)"
    target_file="${XFULL_VICTIM_MODEL}_${lookup}_dog-bird.json"
    [ -s "$ROOT/target_sets/$target_file" ] || \
        die "pinned target set missing: $ROOT/target_sets/$target_file"
    rsync -a "$ROOT/target_sets/$target_file" "$RUN_ROOT/target_sets/"

    [ -d "$DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 data missing: $DATA_ROOT/cifar-10-batches-py"
    rsync -a "$DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"

    for model in "$XFULL_VICTIM_MODEL" "$XFULL_SELECTOR_MODEL"; do
        cache_name="${model}_60ep_lr0.1_bs128_seed42"
        stage_dir_if_present "$CACHE_ROOT/surrogates/$cache_name" \
                             "$LOCAL_CACHE_ROOT/surrogates/$cache_name"
        [ "$model" = "$XFULL_VICTIM_MODEL" ] || continue
        cache_name="${model}_${XFULL_VICTIM_EPOCHS}ep_lr0.1_bs125_wd0_seed42"
        stage_dir_if_present "$CACHE_ROOT/clean_victims/$cache_name" \
                             "$LOCAL_CACHE_ROOT/clean_victims/$cache_name"
    done

    # Dedicated result root prevents the old 5x4 K=1/K=3 trials from being
    # interpreted as completed trials for this 10x6 rerun.
    stage_dir_if_present "$RESULT_ROOT/$XFULL_RUN_NAME" \
                         "$LOCAL_RESULT_ROOT/$XFULL_RUN_NAME"
}

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    say "sync: $XFULL_RUN_NAME -> $RESULT_ROOT"
    if [ -d "$LOCAL_RESULT_ROOT/$XFULL_RUN_NAME" ]; then
        mkdir -p "$RESULT_ROOT/$XFULL_RUN_NAME"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$LOCAL_RESULT_ROOT/$XFULL_RUN_NAME/" \
            "$RESULT_ROOT/$XFULL_RUN_NAME/"
    fi
    # Normally these are already complete because ensure_model_cache ran first;
    # retain this for safe recovery if final_update creates a missing file.
    local model cache_name
    for model in "$XFULL_VICTIM_MODEL" "$XFULL_SELECTOR_MODEL"; do
        cache_name="${model}_60ep_lr0.1_bs128_seed42"
        sync_cache_dir "$LOCAL_CACHE_ROOT/surrogates/$cache_name" \
                       "$CACHE_ROOT/surrogates/$cache_name"
        [ "$model" = "$XFULL_VICTIM_MODEL" ] || continue
        cache_name="${model}_${XFULL_VICTIM_EPOCHS}ep_lr0.1_bs125_wd0_seed42"
        sync_cache_dir "$LOCAL_CACHE_ROOT/clean_victims/$cache_name" \
                       "$CACHE_ROOT/clean_victims/$cache_name"
    done
    say "sync: complete"
}

handle_signal() {
    say "signal: received $1; stopping the active step before final sync"
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
    local path="$1" count="$2"
    python - "$path" "$count" <<'PY'
import json
import sys

path, count = sys.argv[1], int(sys.argv[2])
with open(path) as handle:
    payload = json.load(handle)
try:
    indices = payload['pairs']['dog-bird']['indices']
except (KeyError, TypeError):
    if isinstance(payload, list):
        indices = payload
    else:
        raise SystemExit('unrecognized target-set structure: %s' % path)
indices = list(map(int, indices[:count]))
if len(indices) != count or len(set(indices)) != count:
    raise SystemExit('%s must supply at least %d unique dog-bird targets' %
                     (path, count))
print('targets: %d unique pinned indices:' % count, ' '.join(map(str, indices)))
PY
}

verify_results() {
    local csv_path="$LOCAL_RESULT_ROOT/$XFULL_RUN_NAME/results.csv"
    local lookup target_file expected_rows
    lookup="$(target_attack)"
    target_file="$RUN_ROOT/target_sets/${XFULL_VICTIM_MODEL}_${lookup}_dog-bird.json"
    expected_rows=$((XFULL_NUM_TARGETS * XFULL_NUM_VICTIMS))
    python - "$csv_path" "$target_file" "$XFULL_NUM_TARGETS" \
        "$XFULL_NUM_VICTIMS" "$expected_rows" <<'PY'
import csv
import json
import sys
from collections import Counter

csv_path, target_path = sys.argv[1:3]
num_targets, num_victims, expected_rows = map(int, sys.argv[3:])
with open(target_path) as handle:
    expected = list(map(int, json.load(handle)['pairs']['dog-bird']['indices']))[:num_targets]
with open(csv_path, newline='') as handle:
    rows = list(csv.DictReader(handle))

pairs = []
for row in rows:
    try:
        target = int(row['target_idx'])
        victim = int(row['victim_id'])
        success = int(row['success'])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit('malformed results row: %s' % exc)
    if success not in (0, 1):
        raise SystemExit('non-binary success value for target %d victim %d' % (target, victim))
    pairs.append((target, victim))

counts = Counter(t for t, _ in pairs)
victims = {t: sorted(v for tt, v in pairs if tt == t) for t in counts}
problems = []
if len(rows) != expected_rows:
    problems.append('expected %d rows, found %d' % (expected_rows, len(rows)))
if len(set(pairs)) != expected_rows:
    problems.append('expected %d unique (target,victim) pairs, found %d' %
                    (expected_rows, len(set(pairs))))
if set(counts) != set(expected):
    problems.append('target IDs differ from the pinned set')
for target in expected:
    if victims.get(target) != list(range(num_victims)):
        problems.append('target %d victim IDs are %s, expected 0..%d' %
                        (target, victims.get(target, []), num_victims - 1))
if problems:
    raise SystemExit('incomplete result: ' + '; '.join(problems))
print('verified: %d targets x %d victims = %d unique evaluations' %
      (num_targets, num_victims, expected_rows))
PY
}

main() {
    local required expected_degree lookup target_file status
    local memory_args=() sharp_args=() selector_args=()
    [ -n "${SLURM_TMPDIR:-}" ] || die "SLURM_TMPDIR is unset; submit this with sbatch"
    for required in XFULL_INDEX XFULL_ATTACK XFULL_BUDGET XFULL_VICTIM_MODEL \
                    XFULL_SELECTOR_MODEL XFULL_K XFULL_TARGET_DEGREE \
                    XFULL_RUN_NAME ORIGINAL_COMMAND; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done
    case "$XFULL_ATTACK" in fc|gradmatch|sapa) ;; *) die "bad attack: $XFULL_ATTACK" ;; esac
    case "$XFULL_BUDGET" in 0.002|0.005) ;; *) die "bad budget: $XFULL_BUDGET" ;; esac
    case "$XFULL_VICTIM_MODEL" in ConvNetBN|ResNet20BN|VGG13BN) ;;
        *) die "bad victim model: $XFULL_VICTIM_MODEL" ;;
    esac
    case "$XFULL_SELECTOR_MODEL" in ConvNetBN|ResNet20BN|VGG13BN) ;;
        *) die "bad selector model: $XFULL_SELECTOR_MODEL" ;;
    esac
    case "$XFULL_K" in 1|3|10|20|30) ;; *) die "bad selector K: $XFULL_K" ;; esac
    case "$XFULL_NUM_TARGETS" in 8|10) ;;
        *) die "XFULL_NUM_TARGETS must be 8 or 10 (got $XFULL_NUM_TARGETS)" ;;
    esac
    [ "$XFULL_NUM_VICTIMS" = 6 ] || \
        die "XFULL_NUM_VICTIMS must be 6 (got $XFULL_NUM_VICTIMS)"
    case "$XFULL_BASE_DIST" in
        l2|cosine|cosine_norm) ;;
        *) die "bad base distance: $XFULL_BASE_DIST" ;;
    esac
    case "$XFULL_LAMBDA_MARGIN" in
        1|100) ;;
        *) die "XFULL_LAMBDA_MARGIN must be 1 or 100 (got $XFULL_LAMBDA_MARGIN)" ;;
    esac
    case "$XFULL_VICTIM_EPOCHS:$XFULL_VICTIM_DECAY" in
        50:40|70:50) ;;
        *) die "unsupported victim schedule: epochs=$XFULL_VICTIM_EPOCHS decay=$XFULL_VICTIM_DECAY" ;;
    esac
    case "$XFULL_COMPONENT" in
        '') ;;
        minus-m) selector_args=(--sel_component minus-m --jacobian_batch_size 64) ;;
        *) die "unsupported XFULL_COMPONENT: $XFULL_COMPONENT" ;;
    esac

    case "$XFULL_VICTIM_MODEL:$XFULL_ATTACK" in
        ConvNetBN:fc) expected_degree=50 ;;
        ConvNetBN:gradmatch|ConvNetBN:sapa) expected_degree=70 ;;
        ResNet20BN:fc) expected_degree=10 ;;
        ResNet20BN:gradmatch|ResNet20BN:sapa) expected_degree=14 ;;
        VGG13BN:fc) expected_degree=3 ;;
        VGG13BN:gradmatch|VGG13BN:sapa) expected_degree=50 ;;
    esac
    [ "$XFULL_TARGET_DEGREE" = "$expected_degree" ] || \
        die "target-degree mismatch: got $XFULL_TARGET_DEGREE, expected $expected_degree"

    if command -v module >/dev/null 2>&1; then
        module load python/3.11.5 cuda/12.6 cudnn
    fi
    command -v flock >/dev/null 2>&1 || die "flock is required for shared cache coordination"
    [ -f "$ENV_ACTIVATE" ] || die "environment activation missing: $ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
    python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

    trap 'handle_signal USR1' USR1
    trap 'handle_signal TERM' TERM
    trap 'handle_signal INT' INT
    trap sync_outputs EXIT

    [ -f "$ROOT/final_update.py" ] || die "missing $ROOT/final_update.py"
    [ -d "$DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 data missing: $DATA_ROOT/cifar-10-batches-py"

    # Complete and freeze the common checkpoint pools before copying anything to
    # node-local storage. The selection architecture only needs surrogates; V also
    # needs the six shared clean-victim checkpoints.
    ensure_model_cache "$XFULL_VICTIM_MODEL" 1
    if [ "$XFULL_SELECTOR_MODEL" != "$XFULL_VICTIM_MODEL" ]; then
        ensure_model_cache "$XFULL_SELECTOR_MODEL" 0
    fi

    mkdir -p "$RESULT_ROOT/.locks"
    exec {RUN_LOCK_FD}>"$RESULT_ROOT/.locks/${XFULL_RUN_NAME}.lock"
    say "run: waiting for duplicate-submission lock"
    flock -x "$RUN_LOCK_FD"

    stage_inputs
    lookup="$(target_attack)"
    target_file="$RUN_ROOT/target_sets/${XFULL_VICTIM_MODEL}_${lookup}_dog-bird.json"
    verify_target_file "$target_file" "$XFULL_NUM_TARGETS"

    if [ "$XFULL_VICTIM_MODEL" = VGG13BN ] && \
       { [ "$XFULL_ATTACK" = gradmatch ] || [ "$XFULL_ATTACK" = sapa ]; }; then
        memory_args=(--craft_lowmem --craft_batch 256 --fast_gradmatch)
    fi
    if [ "$XFULL_ATTACK" = sapa ]; then
        sharp_args=(--sharp_mode worst --sharp_sigma 0.05)
    fi

    say "job: ${SLURM_JOB_ID:-unknown} ${SLURM_JOB_NAME:-unknown} on $(hostname)"
    say "config: $ORIGINAL_COMMAND"
    say "protocol: method=ours Jacobian=off targets=$XFULL_NUM_TARGETS victims=$XFULL_NUM_VICTIMS selector_K=$XFULL_K component=${XFULL_COMPONENT:-basis}"
    say "protocol: base_dist=$XFULL_BASE_DIST lambda_margin=$XFULL_LAMBDA_MARGIN victim_epochs=$XFULL_VICTIM_EPOCHS victim_decay=$XFULL_VICTIM_DECAY"
    say "protocol: 30 shared surrogates available; crafting always uses V checkpoints 0..4"
    say "output: $RESULT_ROOT/$XFULL_RUN_NAME"

    run_tracked srun --ntasks=1 python "$RUN_ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$LOCAL_DATA_ROOT" --seed 42 --gpus all \
        --cache_dir "$LOCAL_CACHE_ROOT" --out_dir "$LOCAL_RESULT_ROOT" \
        --model "$XFULL_VICTIM_MODEL" --sel_model "$XFULL_SELECTOR_MODEL" \
        --attack "$XFULL_ATTACK" --base ours \
        --base_dist "$XFULL_BASE_DIST" --lambda_margin "$XFULL_LAMBDA_MARGIN" \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$XFULL_BUDGET" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 --restarts 8 --fc_restarts 1 \
        --craft_ensemble 5 --craft_aug "${memory_args[@]}" "${sharp_args[@]}" \
        "${selector_args[@]}" \
        --num_surrogates 30 --surrogate_epochs 60 --surrogate_lr 0.1 \
        --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
        --sel_K "$XFULL_K" \
        --num_targets "$XFULL_NUM_TARGETS" --target_select "$XFULL_TARGET_DEGREE" \
        --target_idx_file "$target_file" --rank_on_victims \
        --num_victims "$XFULL_NUM_VICTIMS" --victim_epochs "$XFULL_VICTIM_EPOCHS" \
        --victim_lr 0.1 --victim_bs 125 \
        --victim_decay "$XFULL_VICTIM_DECAY" --victim_wd 0 --clean_baseline
    status=$?
    if [ "$status" -eq 0 ]; then
        verify_results
    fi
    sync_outputs
    trap - EXIT
    exit "$status"
}

main "$@"
