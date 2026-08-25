#!/usr/bin/env bash
#
# Resume the unfinished SAPA / DPP run:
#
#   ResNet20BN / sapa (worst, sigma=0.05) / dpp base / frog-airplane
#   budget 0.04 / tgt10
#
# It was killed on the wall clock, not on an error, so everything already on
# disk stays. Fills this cell of table.tex:
#
#     ResNet20 & frog-airplane & SAPA & dpp & .. & .. & .. & .. & .. & <4e-2>
#
# Every flag is copied verbatim from the "args:" line the run logged, so the
# resumed run is bit-identical to the one that died -- note --base ours with
# --sel_dpp --sel_alpha 2.0 (the dpp arm), --sharp_mode worst --sharp_sigma 0.05
# (what makes it sapa rather than gradmatch).
#
# Targets are pinned to the gradmatch set for this combo, which is what keeps
# sapa and gradmatch comparable on identical images.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: the run merges
# its own results_rank*.csv into results.csv, reads poison_cache/, and restarts
# at the first (target, victim) trial that is missing.
#
# Work left, counted from results*.csv and poison_cache/ on disk:
#   39/60 trials done, 10 of 10 targets crafted
#   -> 3 craft(s) x 4544 s + 21 trial(s) x 128 s ~= 4.5 h
#      (both rates measured in this run's own log; b0.04 means 2000 poisons,
#      so the crafts dominate)
#
# Ask for 6 h.

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "rs2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_ResNet20BN_sapa_ours_frog-airplane_b0.04_eps8_seed42_lam1_cosine_seldpp2_worst0.05_ce5_tgt10
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== rs2 | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
    exit 0
fi

echo "=== rs2 | sapa dpp | ResNet20BN / frog-airplane / budget 0.04 ==="
python final_update.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --model ResNet20BN --attack sapa --base ours \
    --class_pair frog-airplane --pair_order poison-target \
    --budget 0.04 --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 \
    --base_dist cosine --lambda_margin 1.0 \
    --sel_dpp --sel_alpha 2.0 \
    --sharp_mode worst --sharp_sigma 0.05 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_targets 10 --target_select 10 \
    --target_idx_file "target_sets/ResNet20BN_gradmatch_frog-airplane.json" \
    --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 \
    --clean_baseline

echo "=== rs2.sh finished ==="
