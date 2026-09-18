#!/usr/bin/env bash
# Run the eight remaining component-ablation configurations on one 8-GPU host.
#
# Each original configuration is assigned to one physical GPU and split into
# two concurrent four-target workers on that GPU.  When both workers finish,
# their 8 x 5 victim evaluations, poison caches, overhead records, and logs are
# merged into the canonical ours_result/<run-name> directory.  Thus downstream
# table scripts see the same 40-row result they would have seen from one job.

set -Eeuo pipefail

ROOT="${ROOT:-/home/ubuntu/PoisonBase}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/ours_result}"
WORK_ROOT="${WORK_ROOT:-$ROOT/.local_bench_shards}"
LOG_ROOT="${LOG_ROOT:-$ROOT/local_logs/bench_missing_8gpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_CACHE="${BOOTSTRAP_CACHE:-1}"

say() { printf '%s\n' "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

[[ -f "$ROOT/final_update.py" ]] || die "missing $ROOT/final_update.py"
[[ -d "$DATA_ROOT/cifar-10-batches-py" ]] || \
    die "missing CIFAR-10 data at $DATA_ROOT/cifar-10-batches-py"

# Prefer an explicitly supplied environment, then the two likely local paths.
if [[ -n "${ENV_ACTIVATE:-}" ]]; then
    [[ -f "$ENV_ACTIVATE" ]] || die "ENV_ACTIVATE does not exist: $ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
elif [[ -f /home/ubuntu/ENV/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /home/ubuntu/ENV/bin/activate
elif [[ -f "$ROOT/ENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/ENV/bin/activate"
else
    say "environment: using the current Python environment ($PYTHON_BIN)"
fi

cd "$ROOT"
mkdir -p "$CACHE_ROOT" "$RESULT_ROOT" "$WORK_ROOT/targets" "$LOG_ROOT/cache"

GPU_COUNT="$($PYTHON_BIN - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || die "could not determine the CUDA device count"
(( GPU_COUNT >= 8 )) || die "need 8 visible CUDA GPUs; PyTorch sees $GPU_COUNT"
say "GPUs: using physical devices 0-7 (two attack workers per GPU)"

# ---------------------------------------------------------------------------
# Cache bootstrap.  The cluster runs reused 20 surrogate and 5 clean-victim
# checkpoints for each architecture.  If they were not copied to this host,
# train only the missing checkpoint IDs, one process per GPU per wave, before
# starting the 16 attack workers.  Set BOOTSTRAP_CACHE=0 to require preexisting
# caches instead.
# ---------------------------------------------------------------------------

declare -a CACHE_TASK_MODEL=()
declare -a CACHE_TASK_PART=()
declare -a CACHE_TASK_ID=()

for model in ConvNetBN ResNet20BN; do
    surrogate_dir="$CACHE_ROOT/surrogates/${model}_60ep_lr0.1_bs128_seed42"
    victim_dir="$CACHE_ROOT/clean_victims/${model}_50ep_lr0.1_bs125_wd0_seed42"
    for ((id = 0; id < 20; id++)); do
        if [[ ! -s "$surrogate_dir/net_${id}.pt" ]]; then
            CACHE_TASK_MODEL+=("$model")
            CACHE_TASK_PART+=(surrogate)
            CACHE_TASK_ID+=("$id")
        fi
    done
    for ((id = 0; id < 5; id++)); do
        if [[ ! -s "$victim_dir/net_${id}.pt" ]]; then
            CACHE_TASK_MODEL+=("$model")
            CACHE_TASK_PART+=(victim)
            CACHE_TASK_ID+=("$id")
        fi
    done
done

precompute_one() {
    local gpu="$1" model="$2" part="$3" id="$4"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$DATA_ROOT" --seed 42 \
        --cache_dir "$CACHE_ROOT" --out_dir "$WORK_ROOT/precompute-output" \
        --model "$model" --gpus 0 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
        --surrogate_bs 128 --surrogate_decay 35 45 \
        --num_victims 5 --victim_epochs 50 --victim_lr 0.1 \
        --victim_bs 125 --victim_decay 40 --victim_wd 0.0 \
        --precompute_only --precompute_part "$part" --precompute_id "$id"
}

if (( ${#CACHE_TASK_ID[@]} > 0 )); then
    if [[ "$BOOTSTRAP_CACHE" != 1 ]]; then
        die "${#CACHE_TASK_ID[@]} cached checkpoints are missing; copy them or rerun with BOOTSTRAP_CACHE=1"
    fi
    say "cache: ${#CACHE_TASK_ID[@]} checkpoint(s) missing; training them in 8-GPU waves"
    for ((start = 0; start < ${#CACHE_TASK_ID[@]}; start += 8)); do
        declare -a wave_pids=()
        declare -a wave_labels=()
        for ((slot = 0; slot < 8 && start + slot < ${#CACHE_TASK_ID[@]}; slot++)); do
            task=$((start + slot))
            model="${CACHE_TASK_MODEL[$task]}"
            part="${CACHE_TASK_PART[$task]}"
            id="${CACHE_TASK_ID[$task]}"
            label="${model}_${part}_${id}"
            say "cache: GPU $slot <- $label"
            precompute_one "$slot" "$model" "$part" "$id" \
                >"$LOG_ROOT/cache/${label}.out" 2>&1 &
            wave_pids+=("$!")
            wave_labels+=("$label")
        done
        cache_failed=0
        for idx in "${!wave_pids[@]}"; do
            if ! wait "${wave_pids[$idx]}"; then
                say "cache: FAILED ${wave_labels[$idx]} (see $LOG_ROOT/cache/${wave_labels[$idx]}.out)" >&2
                cache_failed=1
            else
                say "cache: finished ${wave_labels[$idx]}"
            fi
        done
        (( cache_failed == 0 )) || die "cache bootstrap failed"
    done
else
    say "cache: all ConvNetBN/ResNet20BN surrogate and victim checkpoints are present"
fi

# Make two disjoint four-target files from the first eight pinned targets used
# by the original component-ablation jobs. SAPA intentionally shares GM targets.
make_target_shards() {
    local source_file="$1" prefix="$2"
    "$PYTHON_BIN" - "$source_file" \
        "$WORK_ROOT/targets/${prefix}_part1.json" \
        "$WORK_ROOT/targets/${prefix}_part2.json" <<'PY'
import json
import os
import sys

source, first, second = sys.argv[1:]
with open(source) as handle:
    blob = json.load(handle)
indices = blob['pairs']['dog-bird']['indices'] if 'pairs' in blob else blob['dog-bird']
indices = [int(value) for value in indices[:8]]
if len(indices) != 8 or len(set(indices)) != 8:
    raise SystemExit('%s does not contain eight unique dog-bird targets' % source)
for path, values in ((first, indices[:4]), (second, indices[4:])):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump({'pairs': {'dog-bird': {'indices': values}}}, handle, indent=2)
        handle.write('\n')
    os.replace(temporary, path)
print('%s -> %s | %s' % (source, indices[:4], indices[4:]))
PY
}

CONV_TARGET_SOURCE="$ROOT/target_sets/ConvNetBN_gradmatch_dog-bird.json"
R20_TARGET_SOURCE="$ROOT/target_sets/ResNet20BN_gradmatch_dog-bird.json"
[[ -s "$CONV_TARGET_SOURCE" ]] || die "missing $CONV_TARGET_SOURCE"
[[ -s "$R20_TARGET_SOURCE" ]] || die "missing $R20_TARGET_SOURCE"
make_target_shards "$CONV_TARGET_SOURCE" conv
make_target_shards "$R20_TARGET_SOURCE" r20

# GPU assignment: one original configuration per GPU, two target shards each.
CFG_IDS=(
    bmc02_conv_sa_b0002_m
    bmc05_r20_gm_b0005_m
    bmc07_r20_sa_b0002_r
    bmc08_r20_sa_b0002_m
    bmc09_r20_sa_b0002_gigt
    bmc10_r20_sa_b0005_r
    bmc11_r20_sa_b0005_m
    bmc12_r20_sa_b0005_gigt
)
MODELS=(
    ConvNetBN ResNet20BN ResNet20BN ResNet20BN
    ResNet20BN ResNet20BN ResNet20BN ResNet20BN
)
ATTACKS=(sapa gradmatch sapa sapa sapa sapa sapa sapa)
BUDGETS=(0.002 0.005 0.002 0.002 0.002 0.005 0.005 0.005)
SELECTORS=(minus-m minus-m r minus-m exact r minus-m exact)
DEGREES=(70 14 14 14 14 14 14 14)
TARGET_PREFIXES=(conv r20 r20 r20 r20 r20 r20 r20)
TARGET_SOURCES=(
    "$CONV_TARGET_SOURCE" "$R20_TARGET_SOURCE" "$R20_TARGET_SOURCE" "$R20_TARGET_SOURCE"
    "$R20_TARGET_SOURCE" "$R20_TARGET_SOURCE" "$R20_TARGET_SOURCE" "$R20_TARGET_SOURCE"
)

selector_suffix() {
    case "$1" in
        r) printf '%s' selR ;;
        minus-m) printf '%s' selMinusM ;;
        exact) printf '%s' selexactgigt ;;
        *) die "unknown selector: $1" ;;
    esac
}

canonical_run_name() {
    local model="$1" attack="$2" budget="$3" selector="$4" degree="$5"
    local name="CIFAR10_${model}_${attack}_ours_dog-bird_b${budget}_eps8_seed42_lam1_cosine_$(selector_suffix "$selector")"
    [[ "$attack" == sapa ]] && name+="_worst0.05"
    name+="_ce5_tgt${degree}"
    printf '%s' "$name"
}

result_is_complete() {
    local results_csv="$1" target_source="$2"
    [[ -s "$results_csv" ]] || return 1
    "$PYTHON_BIN" - "$results_csv" "$target_source" <<'PY'
import csv
import json
import sys
from collections import Counter

csv_path, target_path = sys.argv[1:]
with open(target_path) as handle:
    blob = json.load(handle)
targets = blob['pairs']['dog-bird']['indices'] if 'pairs' in blob else blob['dog-bird']
targets = [int(value) for value in targets[:8]]
with open(csv_path, newline='') as handle:
    rows = list(csv.DictReader(handle))
pairs = [(int(row['target_idx']), int(row['victim_id'])) for row in rows
         if row.get('target_idx', '') != '' and row.get('victim_id', '') != '']
expected = {(target, victim) for target in targets for victim in range(5)}
sys.exit(0 if len(rows) == 40 and len(pairs) == 40 and set(pairs) == expected
         and all(count == 1 for count in Counter(pairs).values()) else 1)
PY
}

run_shard() {
    local gpu="$1" model="$2" attack="$3" budget="$4" selector="$5"
    local degree="$6" target_file="$7" shard_out="$8"
    local -a selector_args=()
    local -a sharp_args=()

    case "$selector" in
        r) selector_args=(--sel_component r --jacobian_batch_size 64) ;;
        minus-m) selector_args=(--sel_component minus-m --jacobian_batch_size 64) ;;
        exact) selector_args=(--sel_exact_alignment --jacobian_batch_size 64) ;;
        *) die "unknown selector: $selector" ;;
    esac
    [[ "$attack" == sapa ]] && sharp_args=(--sharp_mode worst --sharp_sigma 0.05)

    mkdir -p "$shard_out"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$DATA_ROOT" --seed 42 \
        --cache_dir "$CACHE_ROOT" --out_dir "$shard_out" \
        --model "$model" --attack "$attack" --base ours \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$budget" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --fc_restarts 1 --craft_ensemble 5 \
        --base_dist cosine --lambda_margin 1.0 \
        "${selector_args[@]}" "${sharp_args[@]}" \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 4 --target_select "$degree" --target_idx_file "$target_file" \
        --num_victims 5 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 --clean_baseline \
        --gpus 0
}

# Merge two successful shard run directories. Publication is atomic for the two
# files table readers consume (results.csv and summary.json). No incomplete
# 20-row result is ever written to the canonical directory.
merge_shards() {
    local part1_run="$1" part2_run="$2" destination="$3" target_source="$4"
    "$PYTHON_BIN" - "$part1_run" "$part2_run" "$destination" "$target_source" <<'PY'
import csv
import datetime
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

parts = [Path(sys.argv[1]), Path(sys.argv[2])]
destination = Path(sys.argv[3])
target_source = Path(sys.argv[4])

with target_source.open() as handle:
    target_blob = json.load(handle)
targets = (target_blob['pairs']['dog-bird']['indices']
           if 'pairs' in target_blob else target_blob['dog-bird'])
targets = [int(value) for value in targets[:8]]
expected = {(target, victim) for target in targets for victim in range(5)}

rows_by_key = {}
fieldnames = None
summaries = []
for part in parts:
    result_path = part / 'results.csv'
    summary_path = part / 'summary.json'
    if not result_path.is_file() or not summary_path.is_file():
        raise SystemExit('missing complete shard output under %s' % part)
    with result_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = fieldnames or reader.fieldnames
        if reader.fieldnames != fieldnames:
            raise SystemExit('CSV schema mismatch between target shards')
        for row in reader:
            key = (int(row['target_idx']), int(row['victim_id']))
            if key in rows_by_key and rows_by_key[key] != row:
                raise SystemExit('conflicting duplicate evaluation %s' % (key,))
            rows_by_key[key] = row
    with summary_path.open() as handle:
        summaries.append(json.load(handle))

got = set(rows_by_key)
if got != expected or len(rows_by_key) != 40:
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    raise SystemExit('refusing incomplete merge: got=%d missing=%s extra=%s'
                     % (len(rows_by_key), missing, extra))

destination.mkdir(parents=True, exist_ok=True)
for directory_name in ('poison_cache', 'overhead'):
    output_directory = destination / directory_name
    output_directory.mkdir(parents=True, exist_ok=True)
    for part in parts:
        input_directory = part / directory_name
        if not input_directory.is_dir():
            continue
        for source in input_directory.iterdir():
            if not source.is_file() or source.name == 'summary.json' or source.name.endswith('.tmp'):
                continue
            shutil.copy2(source, output_directory / source.name)

target_order = {target: position for position, target in enumerate(targets)}
ordered_rows = [rows_by_key[key] for key in sorted(
    rows_by_key, key=lambda item: (target_order[item[0]], item[1]))]
temporary_results = destination / ('results.csv.tmp.%d' % os.getpid())
with temporary_results.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(ordered_rows)
os.replace(temporary_results, destination / 'results.csv')

per_target = defaultdict(list)
clean_test_acc = []
for row in ordered_rows:
    per_target[int(row['target_idx'])].append(int(row['success']))
    clean_test_acc.append(float(row['clean_test_acc']))
target_rates = [sum(per_target[target]) / len(per_target[target]) for target in targets]

summary = dict(summaries[0])
summary['num_targets'] = 8
summary['num_trials'] = 40
summary['asr_mean'] = sum(target_rates) / len(target_rates)
summary['asr_std'] = math.sqrt(sum((value - summary['asr_mean']) ** 2
                                    for value in target_rates) / len(target_rates))
summary['cta_post_mean'] = sum(clean_test_acc) / len(clean_test_acc)
summary['cta_post_std'] = math.sqrt(sum((value - summary['cta_post_mean']) ** 2
                                        for value in clean_test_acc) / len(clean_test_acc))
baseline = summary.get('cta_baseline_mean')
summary['cta_drop_mean'] = (None if baseline is None
                            else summary['cta_post_mean'] - float(baseline))

tallies = [item.get('tally') for item in summaries]
if all(isinstance(tally, list) and len(tally) == len(tallies[0]) for tally in tallies):
    summary['tally'] = [sum(int(tally[index]) for tally in tallies)
                        for index in range(len(tallies[0]))]
    summary['tally_complete'] = sum(summary['tally']) == 40

overhead_records = []
overhead_directory = destination / 'overhead'
if overhead_directory.is_dir():
    for path in sorted(overhead_directory.glob('target_*.json')):
        try:
            with path.open() as handle:
                overhead_records.append(json.load(handle))
        except (OSError, ValueError):
            pass

def values(key):
    return [float(record[key]) for record in overhead_records
            if record.get(key) is not None]

def total(key):
    found = values(key)
    return sum(found) if found else None

def mean(key):
    found = values(key)
    return sum(found) / len(found) if found else None

def maximum(key):
    found = values(key)
    return max(found) if found else None

overhead_summary = {
    'target_records': len(overhead_records),
    'selection_measurements': len(values('selection_wall_seconds')),
    'final_craft_measurements': len(values('final_craft_wall_seconds')),
    'selection_wall_seconds_total': total('selection_wall_seconds'),
    'selection_wall_seconds_mean_per_target': mean('selection_wall_seconds'),
    'selection_cuda_peak_allocated_bytes_max': maximum('selection_cuda_peak_allocated_bytes'),
    'selection_cuda_incremental_peak_allocated_bytes_max': maximum(
        'selection_cuda_incremental_peak_allocated_bytes'),
    'selection_cuda_peak_reserved_bytes_max': maximum('selection_cuda_peak_reserved_bytes'),
    'final_craft_wall_seconds_total': total('final_craft_wall_seconds'),
    'final_craft_wall_seconds_mean_per_target': mean('final_craft_wall_seconds'),
    'final_craft_cuda_peak_allocated_bytes_max': maximum('final_craft_cuda_peak_allocated_bytes'),
    'final_craft_cuda_incremental_peak_allocated_bytes_max': maximum(
        'final_craft_cuda_incremental_peak_allocated_bytes'),
    'final_craft_cuda_peak_reserved_bytes_max': maximum('final_craft_cuda_peak_reserved_bytes'),
    'process_max_rss_kb_max': maximum('process_max_rss_kb'),
    'records': overhead_records,
}
overhead_directory.mkdir(parents=True, exist_ok=True)
with (overhead_directory / 'summary.json').open('w') as handle:
    json.dump(overhead_summary, handle, indent=2, sort_keys=True)

summary.update({
    'overhead_target_records': overhead_summary['target_records'],
    'selection_overhead_measurements': overhead_summary['selection_measurements'],
    'final_craft_overhead_measurements': overhead_summary['final_craft_measurements'],
    'selection_wall_seconds_total': overhead_summary['selection_wall_seconds_total'],
    'selection_wall_seconds_mean_per_target': overhead_summary[
        'selection_wall_seconds_mean_per_target'],
    'selection_cuda_peak_allocated_bytes_max': overhead_summary[
        'selection_cuda_peak_allocated_bytes_max'],
    'selection_cuda_incremental_peak_allocated_bytes_max': overhead_summary[
        'selection_cuda_incremental_peak_allocated_bytes_max'],
    'final_craft_wall_seconds_total': overhead_summary['final_craft_wall_seconds_total'],
    'final_craft_wall_seconds_mean_per_target': overhead_summary[
        'final_craft_wall_seconds_mean_per_target'],
    'final_craft_cuda_peak_allocated_bytes_max': overhead_summary[
        'final_craft_cuda_peak_allocated_bytes_max'],
    'final_craft_cuda_incremental_peak_allocated_bytes_max': overhead_summary[
        'final_craft_cuda_incremental_peak_allocated_bytes_max'],
    'process_max_rss_kb_max': overhead_summary['process_max_rss_kb_max'],
})

temporary_summary = destination / ('summary.json.tmp.%d' % os.getpid())
with temporary_summary.open('w') as handle:
    json.dump(summary, handle, indent=2)
os.replace(temporary_summary, destination / 'summary.json')

with (destination / 'log.txt').open('w') as output:
    for number, part in enumerate(parts, 1):
        output.write('===== four-target shard %d: %s =====\n' % (number, part))
        path = part / 'log.txt'
        if path.is_file():
            output.write(path.read_text(errors='replace'))
        output.write('\n')

manifest = {
    'merged_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_runs': [str(path) for path in parts],
    'target_ids': targets,
    'victims_per_target': 5,
    'rows': 40,
}
with (destination / 'merge_manifest.json').open('w') as handle:
    json.dump(manifest, handle, indent=2)

print('==== %s : MERGED ASR = %.1f%% +/- %.1f%% | 8 targets x 5 victims ===='
      % (destination.name, 100.0 * summary['asr_mean'],
         100.0 * summary['asr_std']))
PY
}

declare -a PIDS=()
declare -a PART_LOGS=()
declare -a ALL_PIDS=()
declare -a SKIP_CONFIG=()

stop_children() {
    trap - INT TERM
    say "signal: terminating local attack workers" >&2
    if (( ${#ALL_PIDS[@]} > 0 )); then
        kill "${ALL_PIDS[@]}" 2>/dev/null || true
        wait "${ALL_PIDS[@]}" 2>/dev/null || true
    fi
    exit 130
}
trap stop_children INT TERM

say "launch: starting the eight configurations"
for index in "${!CFG_IDS[@]}"; do
    cfg="${CFG_IDS[$index]}"
    model="${MODELS[$index]}"
    attack="${ATTACKS[$index]}"
    budget="${BUDGETS[$index]}"
    selector="${SELECTORS[$index]}"
    degree="${DEGREES[$index]}"
    prefix="${TARGET_PREFIXES[$index]}"
    source_targets="${TARGET_SOURCES[$index]}"
    run_name="$(canonical_run_name "$model" "$attack" "$budget" "$selector" "$degree")"
    canonical_dir="$RESULT_ROOT/$run_name"

    if result_is_complete "$canonical_dir/results.csv" "$source_targets"; then
        say "GPU $index: SKIP $cfg; canonical result already has 40/40 evaluations"
        SKIP_CONFIG[$index]=1
        continue
    fi
    SKIP_CONFIG[$index]=0

    for part in 1 2; do
        target_file="$WORK_ROOT/targets/${prefix}_part${part}.json"
        shard_out="$WORK_ROOT/$cfg/part${part}"
        part_log="$LOG_ROOT/${cfg}.part${part}.out"
        say "GPU $index: launch $cfg part $part with $target_file"
        run_shard "$index" "$model" "$attack" "$budget" "$selector" \
            "$degree" "$target_file" "$shard_out" >"$part_log" 2>&1 &
        pid=$!
        process_slot=$((index * 2 + part - 1))
        PIDS[$process_slot]="$pid"
        PART_LOGS[$process_slot]="$part_log"
        ALL_PIDS+=("$pid")
    done
done

failed_configs=0
for index in "${!CFG_IDS[@]}"; do
    [[ "${SKIP_CONFIG[$index]:-0}" == 1 ]] && continue
    cfg="${CFG_IDS[$index]}"
    config_ok=1
    for part in 1 2; do
        process_slot=$((index * 2 + part - 1))
        pid="${PIDS[$process_slot]}"
        if wait "$pid"; then
            say "GPU $index: finished $cfg part $part"
        else
            say "GPU $index: FAILED $cfg part $part; see ${PART_LOGS[$process_slot]}" >&2
            config_ok=0
        fi
    done

    part1_slot=$((index * 2))
    part2_slot=$((index * 2 + 1))
    combined_log="$LOG_ROOT/${cfg}.out"
    combined_tmp="${combined_log}.tmp.$$"
    {
        printf '===== %s: target shard 1 =====\n' "$cfg"
        cat "${PART_LOGS[$part1_slot]}"
        printf '\n===== %s: target shard 2 =====\n' "$cfg"
        cat "${PART_LOGS[$part2_slot]}"
    } >"$combined_tmp"
    mv "$combined_tmp" "$combined_log"

    if (( config_ok == 0 )); then
        failed_configs=$((failed_configs + 1))
        continue
    fi

    model="${MODELS[$index]}"
    attack="${ATTACKS[$index]}"
    budget="${BUDGETS[$index]}"
    selector="${SELECTORS[$index]}"
    degree="${DEGREES[$index]}"
    run_name="$(canonical_run_name "$model" "$attack" "$budget" "$selector" "$degree")"
    part1_run="$WORK_ROOT/$cfg/part1/$run_name"
    part2_run="$WORK_ROOT/$cfg/part2/$run_name"
    canonical_dir="$RESULT_ROOT/$run_name"
    if merge_output="$(merge_shards "$part1_run" "$part2_run" "$canonical_dir" \
        "${TARGET_SOURCES[$index]}" 2>&1)"; then
        printf '\n===== validated merged result =====\n%s\n' "$merge_output" >>"$combined_log"
        say "$merge_output"
        say "GPU $index: merged $cfg; combined stdout: $combined_log"
    else
        printf '\n===== merge failed =====\n%s\n' "$merge_output" >>"$combined_log"
        say "GPU $index: merge FAILED for $cfg; shard data was preserved" >&2
        failed_configs=$((failed_configs + 1))
    fi
done

trap - INT TERM
if (( failed_configs > 0 )); then
    die "$failed_configs configuration(s) failed; rerun this script to resume their shards"
fi

say "done: all requested canonical results are complete"
say "results: $RESULT_ROOT"
say "combined .out files: $LOG_ROOT/*.out"
