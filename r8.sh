#!/usr/bin/env bash
#
# Resume: ResNet20BN / gradmatch / dpp (alpha=2.0) / dog-bird / budget 0.002 / tgt14
#
# One of the three runs left in crash_again.txt; all three died on the Slurm
# wall clock, not on an error, so nothing already computed is thrown away.
#
# Every flag is copied verbatim from the "args:" line the run logged, so the
# resumed run is bit-identical to the one that died -- note --num_surrogates 20,
# --base_dist cosine, --sel_dpp --sel_alpha 2.0, and NO --craft_lowmem /
# --fast_gradmatch (the log confirms craft_lowmem=false, fast_gradmatch=false
# for ResNet20BN/gradmatch).
#
# --no_resume / --recompute_deltas are deliberately NOT passed: the run reads
# its own results_rank*.csv and poison_cache/ and restarts at the first
# (target, victim) trial that is missing.
#
# Work left, measured from results*.csv and poison_cache/ on disk:
#   9/60 trials done -> 51 victim trials over targets 8740 5324 4630 833 2065 7056 9059 6707 9874
#   plus 8 craft(s), at this run's own logged rates  ->  ~2.45 h
#
# r6 / r7 / r8 are independent: run them on three GPUs at once, or chain them
# in one allocation sized for the sum (~4.5 h).

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "r8.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp2_ce5_tgt14
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== r8 | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
    exit 0
fi

echo "=== r8 | dpp alpha=2.0 | ResNet20BN / gradmatch / dog-bird | budget 0.002 ==="
python final_update.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --model ResNet20BN --attack gradmatch --base ours \
    --class_pair dog-bird --pair_order poison-target \
    --budget 0.002 --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_batch 256 \
    --base_dist cosine --lambda_margin 1.0 \
    --sel_dpp --sel_alpha 2.0 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_targets 10 --target_select 14 \
    --target_idx_file "target_sets/ResNet20BN_gradmatch_dog-bird.json" \
    --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 \
    --clean_baseline

echo "=== r8.sh finished ==="
