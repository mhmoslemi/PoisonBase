#!/usr/bin/env bash
#
# Resume: ResNet20BN / sapa (worst, sigma=0.05) / random base / dog-bird
#         budget 0.04 / tgt14
#
# One of the four b0.04 sapa runs killed in crash_again.txt -- all four hit the
# Slurm wall clock together at 14:38-14:41 on 2026-08-16 (jobs 4812176/94/
# 4812208/09), not an error, so the 54 finished trials stay on disk.
#
# Every flag is copied verbatim from the "args:" line the run logged: --base
# random (the random arm, no --sel_*), --sharp_mode worst --sharp_sigma 0.05,
# --num_surrogates 20, and NO --craft_lowmem /
# --fast_gradmatch (the log confirms both false for ResNet20BN).
#
# Targets are pinned to the gradmatch set for this combo, which is what keeps
# sapa paired with gradmatch on identical images.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: the run reads
# its own results_rank*.csv and poison_cache/ and restarts at the first missing
# (target, victim) trial.
#
# Work left, from results*.csv and poison_cache/ on disk:
#   54/60 trials done, 9/10 crafts cached
#   -> the 6 victim trials of target 9874, plus its craft
#   -> 6 x 130 s + 1 x 4540 s ~= 1.48 h
#      (b0.04 = 2000 poisons, so the single craft is ~85% of that)
#      L40S ~1.5 h    H100 ~0.8 h  (at ~1.8x; extrapolated)
#
# Ask for at least 2 h.
#
# sar1..sar4 are independent -- run them on four GPUs at once (~1.5 h wall).

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "sar1.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_ResNet20BN_sapa_random_dog-bird_b0.04_eps8_seed42_worst0.05_ce5_tgt14
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== sar1 | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
    exit 0
fi

echo "=== sar1 | sapa worst sigma=0.05 | ResNet20BN / random / dog-bird | budget 0.04 ==="
python final_update.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --model ResNet20BN --attack sapa --base random \
    --class_pair dog-bird --pair_order poison-target \
    --budget 0.04 --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_batch 256 \
    --base_dist cosine --lambda_margin 1.0 \
    --sharp_mode worst --sharp_sigma 0.05 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_targets 10 --target_select 14 \
    --target_idx_file "target_sets/ResNet20BN_gradmatch_dog-bird.json" \
    --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 \
    --clean_baseline

echo "=== sar1.sh finished ==="
