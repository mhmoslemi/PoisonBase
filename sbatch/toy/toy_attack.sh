#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=toy_poison_attack
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/toy_attack-%j.out
#SBATCH --mail-user=mhmoslemi2338@gmail.com
#SBATCH --mail-type=ALL

# Smoke test only: 1 surrogate x 1 epoch, 2 FC craft steps,
# 1 preselected target, 1 victim x 1 epoch.

set -Eeuo pipefail

SOURCE_ROOT="/home/mmoslem3/scratch/attack_if"
DATA_ROOT="/home/mmoslem3/scratch/data"
PYTHON_ENV="/home/mmoslem3/ENV"
RUN_ROOT="$SLURM_TMPDIR/toy_attack_if"
STEP_PID=""
SYNCED=0

sync_outputs() {
    [ "$SYNCED" = 0 ] || return 0
    SYNCED=1
    echo "sync: toy attack artifacts -> $SOURCE_ROOT"
    if [ -d "$RUN_ROOT/toy_result" ]; then
        mkdir -p "$SOURCE_ROOT/toy_result"
        rsync -a --exclude='.lock' --exclude='*.tmp' \
            "$RUN_ROOT/toy_result/" "$SOURCE_ROOT/toy_result/"
    fi
    if [ -d "$RUN_ROOT/toy_cache" ]; then
        mkdir -p "$SOURCE_ROOT/toy_cache"
        rsync -a --ignore-existing --exclude='*.tmp' \
            "$RUN_ROOT/toy_cache/" "$SOURCE_ROOT/toy_cache/"
    fi
}

handle_signal() {
    echo "signal: stopping toy attack before final sync"
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

mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/target_sets" \
    "$RUN_ROOT/toy_result" "$RUN_ROOT/toy_cache"
for file in final_update.py networks.py utils.py; do
    rsync -a "$SOURCE_ROOT/$file" "$RUN_ROOT/"
done
rsync -a "$DATA_ROOT/cifar-10-batches-py" "$RUN_ROOT/data/"
rsync -a "$SOURCE_ROOT/target_sets/ConvNetBN_fc_dog-bird.json" \
    "$RUN_ROOT/target_sets/"

# Resume a previous toy attempt without staging any paper-result directory.
if [ -d "$SOURCE_ROOT/toy_result" ]; then
    rsync -a --exclude='.lock' --exclude='*.tmp' \
        "$SOURCE_ROOT/toy_result/" "$RUN_ROOT/toy_result/"
fi
if [ -d "$SOURCE_ROOT/toy_cache" ]; then
    rsync -a --exclude='*.tmp' "$SOURCE_ROOT/toy_cache/" "$RUN_ROOT/toy_cache/"
fi

python -c 'import torch; assert torch.cuda.is_available(); print("gpu:", torch.cuda.get_device_name(0))'

srun --ntasks=1 python "$RUN_ROOT/final_update.py" \
    --dataset CIFAR10 --data_path "$RUN_ROOT/data" --seed 42 \
    --cache_dir "$RUN_ROOT/toy_cache" --out_dir "$RUN_ROOT/toy_result" \
    --model ConvNetBN --attack fc --base random \
    --class_pair dog-bird --pair_order poison-target \
    --budget 0.001 --epsilon 0.0313725 \
    --craft_steps 2 --craft_alpha 0.0039216 \
    --restarts 1 --fc_restarts 1 --craft_ensemble 1 \
    --num_surrogates 1 --surrogate_epochs 1 --surrogate_decay 1 \
    --num_targets 1 --target_select 50 \
    --target_idx_file "$RUN_ROOT/target_sets/ConvNetBN_fc_dog-bird.json" \
    --num_victims 1 --victim_epochs 1 --victim_lr 0.1 --victim_bs 125 \
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
