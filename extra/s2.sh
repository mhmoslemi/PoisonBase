#!/usr/bin/env bash
#
# Resume 2 of 2 for the pair of sapa runs killed in crash_again.txt.
#
#   ConvNetBN / sapa / random / frog-airplane / budget 0.005 / tgt35
#
# Job 4806734 hit its wall clock at 21:08:26 while crafting target 3336 (9/10),
# not on an error, so nothing computed needs to be thrown away.
#
# Every flag is copied verbatim from the "args:" line the run logged, so the
# resumed run is bit-identical to the one that died -- note --base random (this
# is the random-base arm, no --sel_* flag), --num_surrogates 20, and
# --sharp_mode worst --sharp_sigma 0.05 which is what makes it sapa rather than
# gradmatch. Targets are pinned to the gradmatch set, which is what makes
# sapa vs gradmatch a paired comparison on identical images.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: the run reads
# its own results_rank0.csv and poison_cache/ and restarts at the first
# (target, victim) trial that is missing.
#
# Work left, measured from results_rank0.csv and poison_cache/ on disk:
#   48/60 trials done, 16/18 crafts cached
#   -> 12 victim trials (targets 3336, 2603) + 2 crafts
#   -> 12 x 50 s + 2 x 243 s ~= 0.30 h
#
# s1.sh is the dog-bird half; the two are independent, so run them on two
# GPUs at once:  sh s1.sh &  sh s2.sh

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# ~0.3 h of GPU work; on a CPU-only node it would silently crawl for days.
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "s2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_ConvNetBN_sapa_random_frog-airplane_b0.005_eps8_seed42_worst0.05_ce5_tgt35
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== s2 | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
    exit 0
fi

echo "=== s2 | sapa worst sigma=0.05 | ConvNetBN / random / frog-airplane | budget 0.005 ==="
python final_update.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --model ConvNetBN --attack sapa --base random \
    --class_pair frog-airplane --pair_order poison-target \
    --budget 0.005 --epsilon 0.0313725 \
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

echo "=== s2.sh finished ==="
