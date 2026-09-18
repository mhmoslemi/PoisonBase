#!/usr/bin/env bash
# Rerun only the four shards that failed with CUDA OOM.

set -u

ROOT="${ROOT:-/home/ubuntu/PoisonBase}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
WORK_ROOT="${WORK_ROOT:-$ROOT/.local_bench_shards}"
LOG_ROOT="${LOG_ROOT:-$ROOT/local_logs/bench_missing_8gpu}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
elif [[ -f /home/ubuntu/unsloth_env/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /home/ubuntu/unsloth_env/bin/activate
fi

mkdir -p "$LOG_ROOT"

run_one() {
    local gpu="$1" cfg="$2" part="$3" attack="$4" selector="$5"
    local out_dir="$WORK_ROOT/$cfg/part$part"
    local target_file="$WORK_ROOT/targets/r20_part${part}.json"
    local -a selector_args sharp_args=()

    if [[ "$selector" == exact ]]; then
        selector_args=(--sel_exact_alignment --jacobian_batch_size 64)
    else
        selector_args=(--sel_component "$selector" --jacobian_batch_size 64)
    fi
    [[ "$attack" == sapa ]] && sharp_args=(--sharp_mode worst --sharp_sigma 0.05)

    echo "GPU $gpu: rerunning $cfg part $part"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT/final_update.py" \
        --dataset CIFAR10 --data_path "$DATA_ROOT" --seed 42 \
        --cache_dir "$CACHE_ROOT" --out_dir "$out_dir" \
        --model ResNet20BN --attack "$attack" --base ours \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --fc_restarts 1 --craft_ensemble 5 \
        --base_dist cosine --lambda_margin 1.0 \
        "${selector_args[@]}" "${sharp_args[@]}" \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 4 --target_select 14 --target_idx_file "$target_file" \
        --num_victims 5 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 --clean_baseline --gpus 0 \
        >>"$LOG_ROOT/${cfg}.part${part}.out" 2>&1
}

declare -a pids=() labels=()

run_one 0 bmc05_r20_gm_b0005_m 1 gradmatch minus-m &
pids+=("$!"); labels+=(bmc05_part1)

run_one 1 bmc10_r20_sa_b0005_r 1 sapa r &
pids+=("$!"); labels+=(bmc10_part1)

run_one 2 bmc11_r20_sa_b0005_m 2 sapa minus-m &
pids+=("$!"); labels+=(bmc11_part2)

run_one 3 bmc12_r20_sa_b0005_gigt 1 sapa exact &
pids+=("$!"); labels+=(bmc12_part1)

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "DONE: ${labels[$i]}"
    else
        echo "FAILED: ${labels[$i]}" >&2
        failed=1
    fi
done

exit "$failed"
