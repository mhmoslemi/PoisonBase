#!/usr/bin/env bash
#
# Resume the one run left in crash_again.txt:
#
#   ConvNetBN / sapa (worst, sigma=0.05) / random base / frog-airplane
#   budget 0.04 / tgt35
#
# Job 4812271 hit its wall clock at 07:45:54 on 2026-08-16, right after starting
# target 2603 (10/10) -- a time-limit kill, not an error, so all nine finished
# targets stay on disk.
#
# Every flag is copied verbatim from the "args:" line the run logged, so the
# resumed run is bit-identical to the one that died -- note --base random (the
# random-base arm, no --sel_* flag), --sharp_mode worst --sharp_sigma 0.05
# (what makes it sapa rather than gradmatch), --num_surrogates 20, and NO
# --craft_lowmem / --fast_gradmatch (the log confirms both false for ConvNetBN).
#
# Targets are pinned to the gradmatch set for this combo, which is what keeps
# sapa and gradmatch paired on identical images.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: the run reads
# its own results_rank*.csv and poison_cache/ and restarts at the first
# (target, victim) trial that is missing.
#
# Work left, measured from results*.csv and poison_cache/ on disk:
#   54/60 trials done -> the 6 victim trials of target 2603, plus its craft
#   (2603 is the one target with no cached poisons)
#   -> 6 x 50 s + 1 x 2181 s ~= 0.69 h   (both rates from this run's own log;
#      b0.04 means 2000 poisons, so the single craft dominates)
#
# Ask for at least 1 h.

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "s5.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_ConvNetBN_sapa_random_frog-airplane_b0.04_eps8_seed42_worst0.05_ce5_tgt35
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== s5 | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
    exit 0
fi

echo "=== s5 | sapa worst sigma=0.05 | ConvNetBN / random / frog-airplane | budget 0.04 ==="
python final_update.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --model ConvNetBN --attack sapa --base random \
    --class_pair frog-airplane --pair_order poison-target \
    --budget 0.04 --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_batch 256 \
    --base_dist cosine --lambda_margin 1.0 \
    --sharp_mode worst --sharp_sigma 0.05 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_targets 10 --target_select 35 \
    --target_idx_file "target_sets/ConvNetBN_gradmatch_frog-airplane.json" \
    --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 \
    --clean_baseline

echo "=== s5.sh finished ==="
