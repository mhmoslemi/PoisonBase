#!/usr/bin/env bash
# Shared runtime for one Gao/FUS baseline experiment.
set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/influence_fus_baselines_result}"
RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/PoisonBase_influence_fus}"
LOCAL_DATA_ROOT="$RUN_ROOT/data"
LOCAL_CACHE_ROOT="$RUN_ROOT/cache"
LOCAL_RESULT_ROOT="$RUN_ROOT/influence_fus_baselines_result"
LOCAL_RUN_DIR=""
PERSISTENT_RUN_DIR=""
METRIC_TAG="CIFAR10_${IFB_MODEL}_60ep_select10_lr0.1_bs128_decay35-45_wd0_seed42/class5"
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

stage_dir_if_present() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        rsync -a --exclude='.lock' --exclude='*.tmp*' "$src/" "$dst/"
    fi
}

cache_has_nets() {
    local directory="$1" count="$2" index
    for ((index = 0; index < count; index++)); do
        [ -s "$directory/net_${index}.pt" ] || return 1
    done
}

sync_output() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    if [ -n "$LOCAL_RUN_DIR" ] && [ -n "$PERSISTENT_RUN_DIR" ] && \
            [ -d "$LOCAL_RUN_DIR" ]; then
        mkdir -p "$PERSISTENT_RUN_DIR"
        rsync -a --exclude='.lock' --exclude='*.tmp*' \
            "$LOCAL_RUN_DIR/" "$PERSISTENT_RUN_DIR/"
    fi
}

handle_signal() {
    say "signal: received $1; stopping active step and syncing partial output"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
        STEP_PID=""
    fi
    sync_output
    trap - EXIT
    exit 143
}

verify_metric_shards() {
    local directory="$1"
    python - "$directory" <<'PY'
import os
import sys
import json
import numpy as np

directory = sys.argv[1]
for index in range(20):
    path = os.path.join(directory, 'net_%d.npz' % index)
    if not os.path.isfile(path):
        raise SystemExit('missing Gao metric shard: ' + path)
    with np.load(path, allow_pickle=False) as blob:
        for key in ('candidate_indices', 'loss', 'gradnorm', 'forgetting'):
            if key not in blob:
                raise SystemExit('%s lacks %s' % (path, key))
            if len(blob[key]) != 5000:
                raise SystemExit('%s:%s has %d rows, expected 5000' %
                                 (path, key, len(blob[key])))
        for key in ('loss', 'gradnorm', 'forgetting'):
            if not np.isfinite(blob[key]).all():
                raise SystemExit('%s:%s contains non-finite values' % (path, key))
    resource_path = os.path.join(directory, 'net_%d.resources.json' % index)
    if not os.path.isfile(resource_path):
        raise SystemExit('missing Gao resource record: ' + resource_path)
    with open(resource_path) as handle:
        resources = json.load(handle)
    required = ('total_wall_seconds', 'trajectory_wall_seconds',
                'trajectory_training_seconds_excluding_metric',
                'metric_eval_wall_seconds', 'cuda_peak_allocated_bytes',
                'cuda_peak_reserved_bytes', 'process_max_rss_kb')
    if any(key not in resources for key in required):
        raise SystemExit('%s lacks resource fields' % resource_path)
    if not all(np.isfinite(float(resources[key])) for key in required):
        raise SystemExit('%s has invalid resource fields' % resource_path)
print('verified: 20 complete Gao metric shards')
PY
}

verify_target_file() {
    local path="$1" expected_count="$2"
    python - "$path" "$expected_count" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    indices = json.load(handle)['pairs']['dog-bird']['indices']
expected_count = int(sys.argv[2])
if len(indices) != expected_count or len(set(map(int, indices))) != expected_count:
    raise SystemExit('target set must contain exactly %d unique IDs' % expected_count)
print('targets:', ' '.join(map(str, indices)))
PY
}

make_target_partition() {
    local source_path="$1" output_path="$2" part_index="$3" part_count="$4"
    python - "$source_path" "$output_path" "$part_index" "$part_count" <<'PY'
import json
import os
import sys

source_path, output_path, part_index, part_count = sys.argv[1:]
part_index, part_count = int(part_index), int(part_count)
with open(source_path) as handle:
    blob = json.load(handle)
indices = list(map(int, blob['pairs']['dog-bird']['indices']))
quotient, remainder = divmod(len(indices), part_count)
start = part_index * quotient + min(part_index, remainder)
stop = start + quotient + (1 if part_index < remainder else 0)
chosen = indices[start:stop]
if not chosen:
    raise SystemExit('empty target partition %d/%d' % (part_index + 1, part_count))
out = {
    '_generated_by': 'influence/FUS four-way target partition',
    '_source': source_path,
    '_part_index_zero_based': part_index,
    '_part_count': part_count,
    'pairs': {'dog-bird': {'indices': chosen}},
}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
temporary = output_path + '.tmp'
with open(temporary, 'w') as handle:
    json.dump(out, handle, indent=2)
os.replace(temporary, output_path)
print(len(chosen))
PY
}

verify_results() {
    local csv_path="$1" target_path="$2" model="$3" expected_targets="$4"
    python - "$csv_path" "$target_path" "$model" "$expected_targets" <<'PY'
import csv
import json
import sys

csv_path, target_path, model, expected_targets = sys.argv[1:]
expected_targets = int(expected_targets)
with open(target_path) as handle:
    expected = set(map(int, json.load(handle)['pairs']['dog-bird']['indices']))
with open(csv_path, newline='') as handle:
    rows = list(csv.DictReader(handle))
pairs = [(int(row['target_idx']), int(row['victim_id'])) for row in rows]
expected_rows = expected_targets * 6
problems = []
if len(expected) != expected_targets:
    problems.append('target file has %d targets, expected %d' %
                    (len(expected), expected_targets))
if len(rows) != expected_rows:
    problems.append('rows=%d, expected %d' % (len(rows), expected_rows))
if len(set(pairs)) != expected_rows:
    problems.append('unique target/victim pairs=%d, expected %d' %
                    (len(set(pairs)), expected_rows))
if {target for target, _ in pairs} != expected:
    problems.append('target IDs differ from pinned set')
if any(row['model'] != model for row in rows):
    problems.append('result model differs from requested model')
for target in expected:
    got = sorted(victim for got_target, victim in pairs if got_target == target)
    if got != list(range(6)):
        problems.append('target %d victim IDs=%s' % (target, got))
if problems:
    raise SystemExit('incomplete result: ' + '; '.join(problems))
print('verified: %d targets x 6 victims = %d unique evaluations' %
      (expected_targets, expected_rows))
PY
}

main() {
    local required file target_file target_path active_target_path cache_name
    local status time_file part_count part_index part_label target_count
    local sharp_args=()
    local time_prefix=()
    [ -n "${SLURM_TMPDIR:-}" ] || die "SLURM_TMPDIR is unset; submit with sbatch"
    for required in IFB_INDEX IFB_SELECTOR IFB_ATTACK IFB_BUDGET IFB_MODEL \
                    IFB_K IFB_TARGET_DEGREE IFB_RUN_NAME ORIGINAL_COMMAND; do
        [ -n "${!required:-}" ] || die "$required is unset"
    done
    case "$IFB_SELECTOR" in gao-loss|gao-gradnorm|gao-forgetting|fus) ;; *) die "bad selector: $IFB_SELECTOR" ;; esac
    case "$IFB_ATTACK" in gradmatch|sapa) ;; *) die "bad attack: $IFB_ATTACK" ;; esac
    case "$IFB_BUDGET" in 0.002|0.005) ;; *) die "bad budget: $IFB_BUDGET" ;; esac
    case "$IFB_MODEL:$IFB_TARGET_DEGREE" in ConvNetBN:70|ResNet20BN:14) ;; *) die "bad model/target degree" ;; esac
    [ "$IFB_K" = 20 ] || die "K must be 20"

    part_count="${IFB_PART_COUNT:-1}"
    part_index="${IFB_PART_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
    if [ "$IFB_SELECTOR" = fus ]; then
        [ "$part_count" = 4 ] || die "FUS must use IFB_PART_COUNT=4"
        case "$part_index" in 0|1|2|3) ;; *) die "FUS part index must be 0..3" ;; esac
    else
        [ "$part_count" = 1 ] || die "only FUS may be partitioned"
        [ "$part_index" = 0 ] || die "non-FUS part index must be 0"
    fi
    if [ "$part_count" = 4 ]; then
        part_label="part_$((part_index + 1))_of_4"
        LOCAL_RESULT_ROOT="$RUN_ROOT/influence_fus_baselines_result/$part_label"
        PERSISTENT_RUN_DIR="$RESULT_ROOT/$IFB_RUN_NAME/parts/$part_label"
    else
        part_label="full"
        PERSISTENT_RUN_DIR="$RESULT_ROOT/$IFB_RUN_NAME"
    fi
    LOCAL_RUN_DIR="$LOCAL_RESULT_ROOT/$IFB_RUN_NAME"

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

    mkdir -p "$RUN_ROOT" "$LOCAL_DATA_ROOT" "$LOCAL_CACHE_ROOT/surrogates" \
        "$LOCAL_CACHE_ROOT/clean_victims" "$LOCAL_CACHE_ROOT/selector_metrics" \
        "$LOCAL_RESULT_ROOT" "$RUN_ROOT/target_sets"
    for file in final_update.py networks.py utils.py; do
        [ -f "$ROOT/$file" ] || die "missing $ROOT/$file"
        rsync -a "$ROOT/$file" "$RUN_ROOT/"
    done
    [ -d "$DATA_ROOT/cifar-10-batches-py" ] || \
        die "CIFAR-10 data missing: $DATA_ROOT/cifar-10-batches-py"
    rsync -a "$DATA_ROOT/cifar-10-batches-py" "$LOCAL_DATA_ROOT/"

    target_file="${IFB_MODEL}_gradmatch_dog-bird.json"
    [ -s "$ROOT/target_sets/$target_file" ] || die "missing target set: $target_file"
    rsync -a "$ROOT/target_sets/$target_file" "$RUN_ROOT/target_sets/"
    target_path="$RUN_ROOT/target_sets/$target_file"
    verify_target_file "$target_path" 10
    active_target_path="$target_path"
    target_count=10
    if [ "$part_count" = 4 ]; then
        active_target_path="$RUN_ROOT/target_sets/${IFB_MODEL}_gradmatch_dog-bird_${part_label}.json"
        target_count="$(make_target_partition \
            "$target_path" "$active_target_path" "$part_index" "$part_count")"
        verify_target_file "$active_target_path" "$target_count"
    fi

    cache_name="${IFB_MODEL}_60ep_lr0.1_bs128_seed42"
    cache_has_nets "$CACHE_ROOT/surrogates/$cache_name" 20 || \
        die "surrogate cache is missing one of net_0.pt..net_19.pt for $IFB_MODEL"
    stage_dir_if_present "$CACHE_ROOT/surrogates/$cache_name" \
                         "$LOCAL_CACHE_ROOT/surrogates/$cache_name"
    cache_name="${IFB_MODEL}_50ep_lr0.1_bs125_wd0_seed42"
    cache_has_nets "$CACHE_ROOT/clean_victims/$cache_name" 6 || \
        die "clean-victim cache is missing one of net_0.pt..net_5.pt for $IFB_MODEL"
    stage_dir_if_present "$CACHE_ROOT/clean_victims/$cache_name" \
                         "$LOCAL_CACHE_ROOT/clean_victims/$cache_name"

    if [[ "$IFB_SELECTOR" == gao-* ]]; then
        stage_dir_if_present "$CACHE_ROOT/selector_metrics/$METRIC_TAG" \
                             "$LOCAL_CACHE_ROOT/selector_metrics/$METRIC_TAG"
        verify_metric_shards "$LOCAL_CACHE_ROOT/selector_metrics/$METRIC_TAG"
    fi
    # This experiment has a dedicated result root, so the first submission is
    # fresh. Staging only its own run directory lets a timed-out job resume at
    # completed target/victim boundaries without importing any older baseline.
    stage_dir_if_present "$PERSISTENT_RUN_DIR" "$LOCAL_RUN_DIR"
    if [ "$IFB_ATTACK" = sapa ]; then
        sharp_args=(--sharp_mode worst --sharp_sigma 0.05)
    fi

    say "job: ${SLURM_JOB_ID:-unknown} ${SLURM_JOB_NAME:-unknown} on $(hostname)"
    say "config: $ORIGINAL_COMMAND"
    say "partition: $part_label ($target_count target(s), 6 victims each)"
    say "output: $PERSISTENT_RUN_DIR"
    mkdir -p "$LOCAL_RUN_DIR"
    time_file="$LOCAL_RUN_DIR/job_gnu_time_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}.txt"
    if [ -x /usr/bin/time ]; then
        time_prefix=(/usr/bin/time -v -o "$time_file")
        say "whole-job CPU/time resources: $time_file"
    else
        say "warning: /usr/bin/time is unavailable; Python phase metrics remain enabled"
    fi

    set +e
    run_tracked srun --ntasks=1 "${time_prefix[@]}" python "$RUN_ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$LOCAL_DATA_ROOT" --seed 42 --gpus all \
        --cache_dir "$LOCAL_CACHE_ROOT" --out_dir "$LOCAL_RESULT_ROOT" \
        --model "$IFB_MODEL" --attack "$IFB_ATTACK" --base ours \
        --base_dist cosine --lambda_margin 1 --sel_criterion "$IFB_SELECTOR" \
        --selector_metric_dir "$LOCAL_CACHE_ROOT/selector_metrics" \
        --sel_metric_epoch 10 --sel_K 20 \
        --fus_iters 10 --fus_alpha 0.5 --fus_search_epochs 50 \
        --fus_search_decay 40 --fus_proxy_steps 0 --fus_proxy_restarts 0 \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$IFB_BUDGET" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 --restarts 8 --fc_restarts 1 \
        --craft_ensemble 5 --craft_aug "${sharp_args[@]}" \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
        --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
        --num_targets "$target_count" --target_select "$IFB_TARGET_DEGREE" \
        --target_idx_file "$active_target_path" --keep_pinned_targets \
        --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0 --clean_baseline
    status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        verify_results "$LOCAL_RUN_DIR/results.csv" "$active_target_path" \
                       "$IFB_MODEL" "$target_count"
    fi
    sync_output
    trap - EXIT
    exit "$status"
}

main "$@"
