#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=toy_poison_defense
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/toy_defense-%j.out
# Smoke test only: replay one previously saved perturbation through FRIENDS,
# using 1 target, 1 victim x 1 epoch, and 1 friendly-noise epoch.

set -Eeuo pipefail

SOURCE_ROOT="/home/mmoslem3/scratch/attack_if"
DATA_ROOT="/home/mmoslem3/scratch/data"
PYTHON_ENV="/home/mmoslem3/ENV"
RUN_ROOT="$SLURM_TMPDIR/toy_defense_if"
ATTACK_RUN="CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp2_ce5_tgt14"
STEP_PID=""
SYNCED=0

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    echo "sync: toy defense artifacts -> $SOURCE_ROOT"
    if [ -d "$RUN_ROOT/toy_defense_result" ]; then
        mkdir -p "$SOURCE_ROOT/toy_defense_result"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$RUN_ROOT/toy_defense_result/" "$SOURCE_ROOT/toy_defense_result/"
    fi
    if [ -d "$RUN_ROOT/toy_cache" ]; then
        mkdir -p "$SOURCE_ROOT/toy_cache"
        rsync -a --ignore-existing --exclude='*.tmp' \
            "$RUN_ROOT/toy_cache/" "$SOURCE_ROOT/toy_cache/"
    fi
}

handle_signal() {
    echo "signal: stopping toy defense before final sync"
    if [ -n "$STEP_PID" ]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
    fi
    sync_outputs
    trap - EXIT
    exit 143
}

trap handle_signal USR1 TERM INT
trap sync_outputs EXIT

module load python/3.11.5 cuda/12.6 cudnn
source "$PYTHON_ENV/bin/activate"

mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/ours_result/$ATTACK_RUN" \
    "$RUN_ROOT/toy_defense_result" "$RUN_ROOT/toy_cache"
for file in defense.py final_update.py networks.py utils.py victim_aug.py; do
    rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
done
rsync -a "$DATA_ROOT/cifar-10-batches-py" "$RUN_ROOT/data/"

# Only the single attack directory whose saved perturbation is replayed.
[ -d "$SOURCE_ROOT/ours_result/$ATTACK_RUN" ] || {
    echo "missing saved attack run: $SOURCE_ROOT/ours_result/$ATTACK_RUN" >&2
    exit 1
}
rsync -a --exclude='.lock' --exclude='*.tmp' \
    "$SOURCE_ROOT/ours_result/$ATTACK_RUN/" "$RUN_ROOT/ours_result/$ATTACK_RUN/"

# Accept the current per-target poison cache or its legacy two-file form.
if ! find "$RUN_ROOT/ours_result/$ATTACK_RUN/poison_cache" \
        -maxdepth 1 -type f -name 'delta_*.pt' -print -quit 2>/dev/null | grep -q .; then
    [ -f "$RUN_ROOT/ours_result/$ATTACK_RUN/deltas.pt" ] && \
    [ -f "$RUN_ROOT/ours_result/$ATTACK_RUN/bases.json" ] || {
        echo "saved attack run has no perturbation tensors to replay" >&2
        exit 1
    }
fi

# Resume an earlier toy-defense attempt, still isolated from defense_result/.
if [ -d "$SOURCE_ROOT/toy_defense_result" ]; then
    rsync -a --exclude='.lock' --exclude='*.tmp' \
        "$SOURCE_ROOT/toy_defense_result/" "$RUN_ROOT/toy_defense_result/"
fi
if [ -d "$SOURCE_ROOT/toy_cache" ]; then
    rsync -a --exclude='*.tmp' "$SOURCE_ROOT/toy_cache/" "$RUN_ROOT/toy_cache/"
fi

python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

srun --ntasks=1 python "$RUN_ROOT/defense.py" \
    --dataset CIFAR10 --data_path "$RUN_ROOT/data" --seed 42 \
    --cache_dir "$RUN_ROOT/toy_cache" \
    --out_dir "$RUN_ROOT/ours_result" \
    --defense_out_dir "$RUN_ROOT/toy_defense_result" \
    --model ResNet20BN --attack gradmatch --base ours \
    --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 \
    --class_pair dog-bird --pair_order poison-target \
    --budget 0.002 --epsilon 0.0313725 \
    --craft_ensemble 5 --target_select 14 \
    --defense friends --friendly_begin_epoch 0 --friendly_epochs 1 \
    --friendly_bs 1024 \
    --num_targets 1 --num_victims 1 \
    --victim_epochs 1 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 1 --victim_wd 0.0 --clean_baseline &
STEP_PID=$!

set +e
wait "$STEP_PID"
status=$?
set -e
STEP_PID=""
sync_outputs
trap - EXIT
exit "$status"
