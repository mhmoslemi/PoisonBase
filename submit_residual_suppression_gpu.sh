#!/usr/bin/env bash
# Submit the GPU portion of tmp.md's residual-suppression experiment on Vulcan.
#
# Every sbatch allocation executes this same file in worker mode and contains
# exactly one Python experiment invocation:
#   10 independent one-checkpoint precompute jobs (5 surrogate + 5 victim),
#    5 independent candidate jobs (one pinned target each), and
#    6 independent ablation jobs (GM/SAPA x full/no-margin/high-margin).
#
# Candidate collection depends on the surrogate caches.  Ablations depend on all
# caches and all five candidate files, so they reuse the exact computed A_i values.
# Re-running is safe: checkpoints/candidate files are reused and attack runs resume.

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
PYTHON_BIN="${PYTHON_BIN:-/home/mmoslem3/ENV/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SOURCE_ROOT/logs-proposiot}"
DATA_PATH="${DATA_PATH:-$SOURCE_ROOT/data}"
TARGET_IDX_FILE="${TARGET_IDX_FILE:-$SOURCE_ROOT/target_sets/ConvNetBN_gradmatch_dog-bird.json}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
GPU_REQUEST="${GPU_REQUEST:-l40s:1}"
MEMORY="${MEMORY:-12G}"
CPUS="${CPUS:-2}"

MODEL="${MODEL:-ConvNetBN}"
CLASS_PAIR="${CLASS_PAIR:-dog-bird}"
SEED="${SEED:-42}"
NUM_TARGETS="${NUM_TARGETS:-5}"
NUM_VICTIMS="${NUM_VICTIMS:-5}"
NUM_SURROGATES="${NUM_SURROGATES:-5}"
TARGET_SELECT="${TARGET_SELECT:-70}"
BUDGET="${BUDGET:-0.002}"
GAMMA="${GAMMA:-1.0}"
BETA="${BETA:-1.0}"
JACOBIAN_BATCH_SIZE="${JACOBIAN_BATCH_SIZE:-64}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-512}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Worker mode.  This branch has one, and only one, Python invocation.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    [ -f "$SOURCE_ROOT/residual_suppression_experiment.py" ] || \
        die "experiment runner missing: $SOURCE_ROOT/residual_suppression_experiment.py"
    [ -x "$PYTHON_BIN" ] || die "Python environment missing: $PYTHON_BIN"
    [ -s "$TARGET_IDX_FILE" ] || die "pinned target file missing: $TARGET_IDX_FILE"
    [ -d "$DATA_PATH/cifar-10-batches-py" ] || \
        die "CIFAR-10 is missing under DATA_PATH=$DATA_PATH"
    mkdir -p "$OUTPUT_ROOT"
    cd "$SOURCE_ROOT"
    export MPLBACKEND=Agg

    common=(
        --output-root "$OUTPUT_ROOT"
        --dataset CIFAR10 --data-path "$DATA_PATH"
        --model "$MODEL" --class-pair "$CLASS_PAIR" --seed "$SEED"
        --num-targets "$NUM_TARGETS" --num-victims "$NUM_VICTIMS"
        --num-surrogates "$NUM_SURROGATES" --target-select "$TARGET_SELECT"
        --target-idx-file "$TARGET_IDX_FILE" --gpus 0
    )
    case "${RS_MODE:-}" in
        precompute)
            mode_args=(precompute "${common[@]}" --kind "$CACHE_KIND" --cache-id "$CACHE_ID")
            ;;
        collect)
            mode_args=(collect "${common[@]}" --target-position "$TARGET_POSITION"
                       --jacobian-batch-size "$JACOBIAN_BATCH_SIZE"
                       --forward-batch-size "$FORWARD_BATCH_SIZE")
            ;;
        ablation)
            mode_args=(ablation "${common[@]}" --attack "$ATTACK" --selector "$SELECTOR"
                       --budget "$BUDGET" --gamma "$GAMMA" --beta "$BETA")
            ;;
        *) die "unknown worker RS_MODE=${RS_MODE:-unset}" ;;
    esac
    exec "$PYTHON_BIN" "$SOURCE_ROOT/residual_suppression_experiment.py" "${mode_args[@]}"
fi

case "$NUM_TARGETS:$NUM_VICTIMS:$NUM_SURROGATES" in
    5:5:5) ;;
    *) die "this paired design is fixed at 5 targets, 5 victims, and 5 surrogates" ;;
esac

mkdir -p "$OUTPUT_ROOT/slurm"

if [ "${DRY_RUN:-0}" != 1 ]; then
    command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable; run this submitter on Vulcan"
    case "$(hostname -s)" in
        vulcan*) ;;
        *) die "run this submitter on Vulcan (current host: $(hostname -s))" ;;
    esac
    [ -f "$SOURCE_ROOT/residual_suppression_experiment.py" ] || \
        die "experiment runner missing under SOURCE_ROOT=$SOURCE_ROOT"
    [ -f "$SOURCE_ROOT/final_update.py" ] || die "final_update.py missing under $SOURCE_ROOT"
    [ -x "$PYTHON_BIN" ] || die "Python environment missing: $PYTHON_BIN"
    [ -s "$TARGET_IDX_FILE" ] || die "pinned target file missing: $TARGET_IDX_FILE"
    [ -d "$DATA_PATH/cifar-10-batches-py" ] || \
        die "CIFAR-10 is missing under DATA_PATH=$DATA_PATH"
fi

submit_one() {
    local name="$1" walltime="$2" dependency="$3" exports="$4"
    local active output job_id
    if [ "${DRY_RUN:-0}" = 1 ]; then
        # Command substitutions run this function in subshells, so derive a stable
        # unique fake id from the job name instead of incrementing a shell counter.
        job_id=$(printf '%s' "$name" | cksum | awk '{print 900000 + ($1 % 90000)}')
        printf 'DRY RUN: sbatch name=%s time=%s dependency=%s export=%s gpu=%s\n' \
            "$name" "$walltime" "${dependency:-none}" "$exports" "$GPU_REQUEST" >&2
        printf '%s\n' "$job_id"
        return
    fi

    active=$(squeue -h -u "${USER:-mmoslem3}" -n "$name" -o '%A' | head -n 1)
    if [ -n "$active" ]; then
        printf 'active: %-25s -> %s\n' "$name" "$active" >&2
        printf '%s\n' "$active"
        return
    fi

    sbatch_args=(
        --parsable --account="$ACCOUNT"
        --job-name="$name" --time="$walltime"
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEMORY"
        --gpus-per-node="$GPU_REQUEST"
        --output="$OUTPUT_ROOT/slurm/%x-%j.out"
        --export="ALL,$exports"
    )
    [ -z "$dependency" ] || sbatch_args+=(--dependency="afterok:$dependency")
    output=$(sbatch "${sbatch_args[@]}" "$SOURCE_ROOT/submit_residual_suppression_gpu.sh")
    job_id=${output%%;*}
    case "$job_id" in *[!0-9]*|'') die "could not parse sbatch output: $output" ;; esac
    printf 'submitted: %-22s -> %s\n' "$name" "$job_id" >&2
    printf '%s\n' "$job_id"
}

join_ids() {
    local joined="" id
    for id in "$@"; do
        [ -n "$id" ] || continue
        joined="${joined:+$joined:}$id"
    done
    printf '%s\n' "$joined"
}

# Reuse the project's normal seed-matched checkpoints when they are already on
# Vulcan, while keeping the experiment self-contained under logs-proposiot.
# Missing files are produced by the one-checkpoint jobs below.
seed_checkpoint() {
    local kind="$1" cache_id="$2" relative src dst
    case "$kind" in
        surrogate)
            relative="surrogates/${MODEL}_60ep_lr0.1_bs128_seed${SEED}/net_${cache_id}.pt"
            ;;
        victim)
            relative="clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed${SEED}/net_${cache_id}.pt"
            ;;
        *) die "unknown checkpoint kind: $kind" ;;
    esac
    src="$SOURCE_ROOT/cache/$relative"
    dst="$OUTPUT_ROOT/cache/$relative"
    [ -s "$dst" ] && return
    if [ "${DRY_RUN:-0}" != 1 ] && [ -s "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -p "$src" "$dst"
        printf 'seeded cached %-9s %d -> %s\n' "$kind" "$cache_id" "$dst" >&2
    fi
}

precompute_ids=()
for cache_id in 0 1 2 3 4; do
    seed_checkpoint surrogate "$cache_id"
    checkpoint="$OUTPUT_ROOT/cache/surrogates/${MODEL}_60ep_lr0.1_bs128_seed${SEED}/net_${cache_id}.pt"
    if [ -s "$checkpoint" ]; then
        printf 'cached: %-25s -> %s\n' "surrogate $cache_id" "$checkpoint" >&2
        continue
    fi
    precompute_ids+=("$(submit_one "rs-pre-s${cache_id}" 01:30:00 "" \
        "RS_MODE=precompute,CACHE_KIND=surrogate,CACHE_ID=$cache_id")")
done
for cache_id in 0 1 2 3 4; do
    seed_checkpoint victim "$cache_id"
    checkpoint="$OUTPUT_ROOT/cache/clean_victims/${MODEL}_50ep_lr0.1_bs125_wd0_seed${SEED}/net_${cache_id}.pt"
    if [ -s "$checkpoint" ]; then
        printf 'cached: %-25s -> %s\n' "victim $cache_id" "$checkpoint" >&2
        continue
    fi
    precompute_ids+=("$(submit_one "rs-pre-v${cache_id}" 01:30:00 "" \
        "RS_MODE=precompute,CACHE_KIND=victim,CACHE_ID=$cache_id")")
done
precompute_dependency="$(join_ids "${precompute_ids[@]}")"

collect_ids=()
for target_position in 0 1 2 3 4; do
    collect_ids+=("$(submit_one "rs-collect-t${target_position}" 02:55:00 \
        "$precompute_dependency" "RS_MODE=collect,TARGET_POSITION=$target_position")")
done
collect_dependency="$(join_ids "${collect_ids[@]}")"
ablation_dependency="$(join_ids "$precompute_dependency" "$collect_dependency")"

for attack in gradmatch sapa; do
    for selector in full no-margin high-margin; do
        short_selector=${selector//-margin/margin}
        short_attack=$attack
        [ "$attack" = gradmatch ] && short_attack=gm
        [ "$attack" = sapa ] && short_attack=sa
        submit_one "rs-${short_attack}-${short_selector}" 02:55:00 \
            "$ablation_dependency" \
            "RS_MODE=ablation,ATTACK=$attack,SELECTOR=$selector" >/dev/null
    done
done

printf '\nGPU pipeline submitted. All caches, results, and SLURM logs go to:\n  %s\n' "$OUTPUT_ROOT"
printf 'After the jobs finish, run directly (not through sbatch):\n  OUTPUT_ROOT=%q sh %q\n' \
    "$OUTPUT_ROOT" "$SOURCE_ROOT/run_residual_suppression_cpu.sh"
