#!/usr/bin/env bash
#
# Resume: ResNet20BN / sapa (worst, sigma=0.05) / random base / frog-airplane
#         budget 0.005 / tgt10
#
# One of the two runs added to crash_again.txt; it died on the Slurm wall clock
# (STEP ... CANCELLED DUE TO TIME LIMIT just after 00:14 on 2026-08-16), not on
# an error, so nothing already computed is thrown away.
#
# Every flag is copied verbatim from the "args:" line the run logged, so the
# resumed run is bit-identical to the one that died -- note --base random (this
# is the random-base arm, no --sel_* flag), --sharp_mode worst --sharp_sigma
# 0.05 (what makes it sapa rather than gradmatch), --num_surrogates 20, and NO
# --craft_lowmem / --fast_gradmatch (the log confirms both false).
#
# Targets are pinned to the gradmatch set for this combo, which is what makes
# sapa vs gradmatch a paired comparison on identical images.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: the run reads
# its own results_rank*.csv and poison_cache/ and restarts at the first
# (target, victim) trial that is missing.
#
# Work left, measured from results*.csv and poison_cache/ on disk:
#   42/60 trials done -> 18 victim trials over targets 2570 3117 2041
#   plus 3 craft(s), at this run's own logged rates (victim ~130 s,
#   craft ~472 s)  ->  ~1.04 h
#
# s3.sh and s4.sh are independent: sh s3.sh & sh s4.sh on two GPUs (~1.1 h),
# or chain them in one allocation of ~2.5 h.

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "s3.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_ResNet20BN_sapa_random_frog-airplane_b0.005_eps8_seed42_worst0.05_ce5_tgt10
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== s3 | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
    exit 0
fi

echo "=== s3 | sapa worst sigma=0.05 | ResNet20BN / random / frog-airplane | budget 0.005 ==="
python final_update.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --model ResNet20BN --attack sapa --base random \
    --class_pair frog-airplane --pair_order poison-target \
    --budget 0.005 --epsilon 0.0313725 \
    --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 --craft_batch 256 \
    --base_dist cosine --lambda_margin 1.0 \
    --sharp_mode worst --sharp_sigma 0.05 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_targets 10 --target_select 10 \
    --target_idx_file "target_sets/ResNet20BN_gradmatch_frog-airplane.json" \
    --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 \
    --clean_baseline

echo "=== s3.sh finished ==="
