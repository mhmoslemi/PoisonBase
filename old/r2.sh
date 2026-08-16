#!/usr/bin/env bash
#
# Resume 2 of 4 -- VGG13BN / gradmatch / dog-bird / budget 0.04 / tgt50
#
# The original run (sel_dpp_grad2_.sh, job 4794911 on kn028) was NOT killed by
# the terminal disconnect at 07:37 -- it kept going and died at 11:46 when the
# allocation hit walltime, mid-craft on target 6033.
#
# State on disk (ours_result/CIFAR10_VGG13BN_gradmatch_ours_dog-bird_b0.04_
# eps8_seed42_lam1_cosine_seldpp2_ce5_tgt50/):
#     results_rank0.csv   42 rows = 7 targets x 6 victims, all ASR=100.0%
#     poison_cache/       14 files
#     targets 630 409 2270 9870 1656 5022 9090  done
#     targets 6033 7940 9731                    REMAINING
#
# Every flag below is copied verbatim from the "args:" line this run logged, so
# the resumed run is bit-identical to the one that died.  This combo IS
# craft_lowmem (craft_lowmem=True, fast_gradmatch=True in the args), hence
# --craft_lowmem --craft_batch 256 --fast_gradmatch below.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: the run reads its
# own results_rank0.csv + poison_cache/ and restarts at the first missing
# (target, victim) trial.  No surrogate or clean victim is retrained.
#
# Cost: crafting is ~4630 s/target and a victim trial ~100 s, so each remaining
# target is ~1.45 h and the whole script is ~4.4 h.  Ask for more than that:
#     salloc --account=aip-boyuwang --time=0-6:00:00 --gpus-per-node=1 \
#            --cpus-per-task=1 --mem=7G
#     sh r2.sh
# The previous 8:20 allocation was not enough for all 10 targets.

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# On a CPU-only node this would silently crawl for days, which is how earlier
# batches were lost.
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "r2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

IDX="target_sets/VGG13BN_gradmatch_dog-bird.json"
[ -s "$IDX" ] || { echo "r2.sh: target file $IDX missing"; exit 1; }

echo "=== resume | dpp alpha=2.0 | VGG13BN / gradmatch / dog-bird | budget 0.04 ==="
echo "    7/10 targets already on disk; resuming at target 6033 (8/10)"
echo

python final_update.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --model VGG13BN --attack gradmatch --base ours \
    --class_pair dog-bird --pair_order poison-target \
    --budget 0.04 --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 \
    --craft_lowmem --craft_batch 256 --fast_gradmatch \
    --base_dist cosine --lambda_margin 1.0 \
    --sel_dpp --sel_alpha 2.0 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_targets 10 --target_select 50 \
    --target_idx_file "$IDX" \
    --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 \
    --clean_baseline

echo "=== r2.sh finished ==="
