#!/usr/bin/env bash
#
# Resume the one run left in check.txt:
#
#   ResNet20BN / gradmatch / dog-bird / budget 0.005
#   bases selected by VGG13BN  (--sel_model, the cross-architecture cell
#   S = VGG13 -> A = V = ResNet20BN of tab:cross-architecture)
#
# Job 4833059 hit its wall clock at 21:27:34 on 2026-08-16, 338 s into the craft
# of its fifth and last target -- a time-limit kill, not an error, so the four
# finished targets stay on disk.
#
# Every flag below is copied verbatim from the "args:" line the run logged, so
# the resumed run is bit-identical to the one that died. Note --sel_model
# VGG13BN (what makes this a cross-architecture cell: the bases come from
# VGG13BN's surrogates, the crafting and the victims are ResNet20BN), --sel_dpp
# --sel_alpha 2.0, and NO --craft_lowmem / --fast_gradmatch (the log confirms
# both false for ResNet20BN/gradmatch).
#
# --no_resume / --recompute_deltas are deliberately NOT passed: the run reads its
# own results_rank*.csv and poison_cache/ and restarts at the first (target,
# victim) trial that is missing.
#
# Work left, from the run's own log:
#   16/20 trials done (targets 833, 2065, 4630, 5118)
#   -> target 5324: 1 craft + 4 victims
#      533 s craft + 4 x 82 s victims ~= 0.24 h   (both rates measured in this
#      run; the craft has to start over, nothing of it was cached)
#
# Ask for 1 h.

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "xa4b.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2_selarchVGG13BN_ce5_tgt14
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== xa4b | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
    exit 0
fi

echo "=== xa4b | cross-arch S=VGG13BN -> A=V=ResNet20BN | gradmatch / dog-bird / budget 0.005 ==="
python final_update.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --model ResNet20BN --sel_model VGG13BN --attack gradmatch --base ours \
    --class_pair dog-bird --pair_order poison-target \
    --budget 0.005 --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 \
    --base_dist cosine --lambda_margin 1.0 \
    --sel_dpp --sel_alpha 2.0 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_targets 5 --target_select 14 \
    --target_idx_file "target_sets/xarch_ResNet20BN_gradmatch_dog-bird_b0.005.json" \
    --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 \
    --clean_baseline

echo "=== xa4b.sh finished ==="
