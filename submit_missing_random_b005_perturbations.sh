#!/bin/bash
#SBATCH --account=aip-bghaddar
#SBATCH --job-name=rand_b005_4delta
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/mmoslem3/PoisonBase/sbatch/logs/%x-%j.out

set -Eeuo pipefail

# Running this file with `sh` submits it; the body runs only inside Slurm.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    exec sbatch "$0"
fi

ROOT=/scratch/mmoslem3/PoisonBase
ENV_ACTIVATE=/home/mmoslem3/ENV/bin/activate
RUN_ROOT="$SLURM_TMPDIR/PoisonBase_missing_random_b005"
LOCAL_DATA="$RUN_ROOT/data"
LOCAL_CACHE="$RUN_ROOT/cache"
LOCAL_RESULTS="$RUN_ROOT/ours_result"
RUN_NAME=CIFAR10_ConvNetBN_gradmatch_random_dog-bird_b0.005_eps8_seed42_K20_ce5_tgt70
PERSIST_RUN="$ROOT/ours_result/$RUN_NAME"
LOCAL_RUN="$LOCAL_RESULTS/$RUN_NAME"
TARGETS=(8954 821 1547 4427)
STEP_PID=""
SYNCED=0

sync_outputs() {
    [ "$SYNCED" -eq 0 ] || return 0
    SYNCED=1
    mkdir -p "$PERSIST_RUN/poison_cache"
    for target in "${TARGETS[@]}"; do
        for prefix in base delta; do
            case "$prefix" in
                base) suffix=json ;;
                delta) suffix=pt ;;
            esac
            src="$LOCAL_RUN/poison_cache/${prefix}_${target}.${suffix}"
            [ -s "$src" ] || continue
            rsync -a --ignore-existing "$src" "$PERSIST_RUN/poison_cache/"
        done
    done
}

stop_and_sync() {
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
        STEP_PID=""
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

trap stop_and_sync USR1 TERM INT
trap sync_outputs EXIT

module load python/3.11.5 cuda/12.6 cudnn
source "$ENV_ACTIVATE"

mkdir -p "$LOCAL_DATA" "$LOCAL_CACHE/surrogates" "$LOCAL_RESULTS" \
         "$RUN_ROOT/target_sets" "$LOCAL_RUN/poison_cache"
rsync -a "$ROOT/final_update.py" "$ROOT/networks.py" "$ROOT/utils.py" "$RUN_ROOT/"
rsync -a "$ROOT/data/cifar-10-batches-py" "$LOCAL_DATA/"
rsync -a "$ROOT/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42" \
          "$LOCAL_CACHE/surrogates/"

# Stage any targets completed by an earlier interrupted attempt.
if [ -d "$PERSIST_RUN/poison_cache" ]; then
    for target in "${TARGETS[@]}"; do
        [ ! -s "$PERSIST_RUN/poison_cache/base_${target}.json" ] || \
            rsync -a "$PERSIST_RUN/poison_cache/base_${target}.json" "$LOCAL_RUN/poison_cache/"
        [ ! -s "$PERSIST_RUN/poison_cache/delta_${target}.pt" ] || \
            rsync -a "$PERSIST_RUN/poison_cache/delta_${target}.pt" "$LOCAL_RUN/poison_cache/"
    done
fi

cat > "$RUN_ROOT/target_sets/missing_random_b005.json" <<'JSON'
{"pairs":{"dog-bird":{"indices":[8954,821,1547,4427]}}}
JSON

echo "job=$SLURM_JOB_ID node=$(hostname) targets=${TARGETS[*]}"
srun --ntasks=1 python "$RUN_ROOT/final_update.py" \
    --dataset CIFAR10 --data_path "$LOCAL_DATA" --seed 42 \
    --cache_dir "$LOCAL_CACHE" --out_dir "$LOCAL_RESULTS" \
    --model ConvNetBN --attack gradmatch --base random \
    --class_pair dog-bird --pair_order poison-target \
    --budget 0.005 --epsilon 0.0313725 \
    --craft_steps 750 --craft_alpha 0.0039216 --restarts 8 \
    --craft_ensemble 5 --sel_K 20 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_lr 0.1 \
    --surrogate_bs 128 --surrogate_decay 35 45 --surrogate_wd 0 \
    --num_targets 4 --target_select 70 \
    --target_idx_file "$RUN_ROOT/target_sets/missing_random_b005.json" \
    --keep_pinned_targets \
    --num_victims 1 --victim_epochs 0 --victim_lr 0.1 \
    --victim_bs 125 --victim_wd 0 &
STEP_PID=$!
wait "$STEP_PID"
STEP_PID=""

for target in "${TARGETS[@]}"; do
    test -s "$LOCAL_RUN/poison_cache/base_${target}.json"
    test -s "$LOCAL_RUN/poison_cache/delta_${target}.pt"
done

sync_outputs
trap - EXIT
echo "saved four perturbations in $PERSIST_RUN/poison_cache"
