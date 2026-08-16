#!/usr/bin/env bash
#
# Resume the 3 runs killed in dpp_crash2.txt -- VGG13BN / fc / dog-bird,
# dpp selection with SEL_ALPHA=2.0.  All three died on the Slurm wall clock
# (STEP ... CANCELLED DUE TO TIME LIMIT at 00:03:49, 00:04:19 and ~00:05:32),
# not on an error, so nothing needs to be thrown away.
#
# These are exactly the three "--" cells in the
#   VGG13 & dog-bird & fc & dpp & 3
# row of table.tex; finishing them completes that row.
#
# Every flag is copied verbatim from the "args:" line each run logged, so the
# resumed run is bit-identical to the one that died -- note --num_surrogates 20
# (NOT the 5 in sel_dpp.sh) and --base_dist cosine --sel_dpp --sel_alpha 2.0.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: each run reads
# its own results_rank*.csv and poison_cache/ and restarts at the first
# (target, victim) trial that is missing.
#
# Work left (counted from results_rank0.csv and poison_cache/ on disk):
#   b0.002    3 victim trials + 0 crafts  (all 10 targets already crafted)  ~0.11 h
#   b0.01     5 victim trials + 0 crafts  (all 10 targets already crafted)  ~0.17 h
#   b0.04    15 victim trials + 2 crafts                                    ~0.49 h
#   -----------------------------------------------------------------------------
#   estimated total: ~0.77 h   (victim ~100 s, craft ~92 s at b0.04, from the log)
#
# Cheapest first, so a short allocation still lands two complete budgets.
# Ask for at least 1 h -- a 4th time-limit kill costs another round trip.

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# ~0.8 h of GPU work; on a CPU-only node it would silently crawl for days.
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "d5.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

run () {   # run <budget>
    echo "=== d5 | dpp alpha=2.0 | VGG13BN / fc / dog-bird | budget $1 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model VGG13BN --attack fc --base ours \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$1" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 --craft_batch 256 \
        --base_dist cosine --lambda_margin 1.0 \
        --sel_dpp --sel_alpha 2.0 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 10 --target_select 3 \
        --target_idx_file "target_sets/VGG13BN_fc_dog-bird.json" \
        --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline
}

run 0.002
run 0.01
run 0.04

echo "=== d5.sh finished ==="
