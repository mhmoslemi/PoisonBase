#!/usr/bin/env bash
#
# Finish the last 4 unfinished runs of the dpp (SEL_ALPHA=2.0) sweep -- the ones
# still sitting in dpp_crash.txt.  All four are fc / ours / frog-airplane.
#
# Every flag below is copied verbatim from the "args:" line each of these runs
# logged, so the resumed run is bit-identical to the one that died -- note
# --num_surrogates 20 (NOT the 5 in sel_dpp.sh) and
# --base_dist cosine --sel_dpp --sel_alpha 2.0.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: each run reads
# its own results_rank*.csv and poison_cache/ and restarts at the first
# (target, victim) trial that is missing.  Nothing already computed is redone,
# and no surrogate or clean victim is retrained (the cache keys are unchanged).
#
# Work left, measured from results*.csv and poison_cache/ on disk at 00:58 on
# 2026-08-15 (victim ~79 s VGG13 / ~82 s ResNet20; craft as logged):
#
#   VGG13BN     b0.001   53/60   7 trials, targets 338 4287, 1 craft   ~0.15 h
#   VGG13BN     b0.002   52/60   8 trials, targets 338 4287, 1 craft   ~0.18 h
#   VGG13BN     b0.01    55/60   5 trials, target  4287,     0 crafts  ~0.11 h
#   ResNet20BN  b0.02    50/60  10 trials, targets 2672 6481, 1 craft  ~0.23 h
#   -------------------------------------------------------------------------
#   estimated total: ~0.68 h        -- comfortably inside a 1 h allocation
#
# b0.001 and b0.002 are the stalled pair (nothing written since ~22:00; b0.002
# was the queued tail of the dead d4.sh job, b0.001 was never in any script),
# so they go first: a short allocation should land them before anything else.
#
# b0.01 and ResNet20 b0.02 were still being written by live jobs when this was
# generated.  The guard below skips any run that has already finished, and
# refuses to touch one whose log.txt was written in the last SKIP_MIN minutes,
# so starting this while those jobs are still going will not double-run them.
# Override with  FORCE=1 sh d6.sh  once you know the other jobs are gone.

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42
SKIP_MIN=${SKIP_MIN:-15}     # treat a run as "live elsewhere" if touched this recently
FORCE=${FORCE:-0}            # FORCE=1 ignores the liveness guard

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# These runs are ~0.7 h of GPU work; on a CPU-only node they would silently
# crawl for days, which is how the earlier batch was lost.
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "d6.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

run () {   # run <model> <target_select> <budget>
    tag="CIFAR10_$1_fc_ours_frog-airplane_b$3_eps8_seed42_lam1_cosine_seldpp2_ce5_tgt$2"
    dir="$OUT_DIR/$tag"

    # already finished -- the summary line is written only after all 60 trials
    if grep -q "==== $tag : ASR = " "$dir/log.txt" 2>/dev/null; then
        echo "=== d6 | SKIP $1 / frog-airplane / b$3 -- already complete:"
        grep -o "ASR = .*====" "$dir/log.txt" | tail -1
        return 0
    fi

    # someone else is still writing it (a live job from the previous batch)
    if [ "$FORCE" != "1" ] && [ -f "$dir/log.txt" ] && \
       [ -n "$(find "$dir/log.txt" -mmin -"$SKIP_MIN" 2>/dev/null)" ]; then
        echo "=== d6 | SKIP $1 / frog-airplane / b$3 -- log.txt touched < $SKIP_MIN min ago;"
        echo "         another job still owns it.  Re-run later, or FORCE=1 sh d6.sh"
        return 0
    fi

    echo "=== d6 | dpp alpha=2.0 | $1 / fc / frog-airplane | budget $3 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model "$1" --attack fc --base ours \
        --class_pair frog-airplane --pair_order poison-target \
        --budget "$3" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 --craft_batch 256 \
        --base_dist cosine --lambda_margin 1.0 \
        --sel_dpp --sel_alpha 2.0 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 10 --target_select "$2" \
        --target_idx_file "target_sets/$1_fc_frog-airplane.json" \
        --num_victims 6 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline
}

# stalled pair first, then the two that live jobs may already have finished
run VGG13BN     3 0.001
run VGG13BN     3 0.002
run VGG13BN     3 0.01
run ResNet20BN  2 0.02

echo "=== d6.sh finished ==="
