#!/usr/bin/env bash
#
# appendix.tex, tab:base-redundancy
#
# tab:base-redundancy -- one row: Random, Top-1, Top-2, Top-5, Top-10, Greedy, DPP.
#
# ConvNetBN / SAPA / dog -> bird (--class_pair dog-bird) at eps=5e-3.  7 runs.
#
# Top-r keeps the r highest-scoring bases and spreads the poison budget over them,
# every copy still optimized independently (--base_topr, added for this table).
# Top-m is plain Greedy, so it is the --base ours run with no --sel_* flag.
#
# PROTOCOL (appendix.tex preamble): ConvNetBN unless stated, 5 targets sampled
# uniformly from the target class with a fixed seed (NOT by difficulty), 5 victims
# with seeds 0-4, selector defaults lambda=1 alpha=2 K=20. Targets are frozen by
# appendix/pin_targets.py before anything runs and shared by every method.
#
# PAIR NOTATION: appendix.tex writes pairs as target -> adversarial. final_update.py
# takes --class_pair "<adv>-<target>" under --pair_order poison-target, so the
# mapping below is the paper's arrow read right-to-left. Note this makes the
# appendix's dog -> bird the OPPOSITE direction from the main sweep's "dog-bird"
# run dirs (those are y_adv=dog, target=bird). That is what the appendix text
# specifies; flip PAIR below if the intent was to match the main sweep instead.
#
#   sh appendix/ap8-redundancy.sh
#   DRY_RUN=1 sh appendix/ap8-redundancy.sh

set -u

DATA_PATH=/home/mmoslem3/scratch/data
SEED=42
NT="${NUM_TARGETS:-5}"
NV="${NUM_VICTIMS:-5}"
DRY_RUN="${DRY_RUN:-}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ap8-redundancy.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

COMMON="--dataset CIFAR10 --data_path $DATA_PATH --seed $SEED \
    --cache_dir ./cache --out_dir ours_result --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets $NT --num_victims $NV"

run() {   # $1 = run tag, $2.. = flags
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "ours_result/$TAG/log.txt" 2>/dev/null; then
        echo "--- already complete: $TAG"; return 0
    fi
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then echo "    python final_update.py $*"; return 0; fi
    python final_update.py "$@" || exit 1
}

pin() {   # $1 = pair (adv-target), $2 = out file, $3 = model
    [ -s "$2" ] && return 0
    [ -n "$DRY_RUN" ] && { echo "    pin_targets $1 -> $2"; return 0; }
    python appendix/pin_targets.py --model "$3" --pair "$1" --target_select random \
        --num_targets "$NT" --num_victims "$NV" --out "$2" || exit 1
}

IDX="target_sets/appx_matched_dog-bird.json"
pin dog-bird "$IDX" ConvNetBN

SH="--sharp_mode worst --sharp_sigma 0.05"

LABEL="redundancy | Random"
run "CIFAR10_ConvNetBN_sapa_random_dog-bird_b0.005_eps8_seed42_worst0.05_ce5" \
    $COMMON --model ConvNetBN --attack sapa --base random --class_pair dog-bird \
    --budget 0.005 $SH --target_idx_file "$IDX"

for R in 1 2 5 10; do
    LABEL="redundancy | Top-$R"
    run "CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_top${R}_worst0.05_ce5" \
        $COMMON --model ConvNetBN --attack sapa --base ours --class_pair dog-bird \
        --base_dist cosine --lambda_margin 1.0 --base_topr $R \
        --budget 0.005 $SH --target_idx_file "$IDX"
done

LABEL="redundancy | Greedy (Top-m)"
run "CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_worst0.05_ce5" \
    $COMMON --model ConvNetBN --attack sapa --base ours --class_pair dog-bird \
    --base_dist cosine --lambda_margin 1.0 --budget 0.005 $SH --target_idx_file "$IDX"

LABEL="redundancy | DPP"
run "CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2_worst0.05_ce5" \
    $COMMON --model ConvNetBN --attack sapa --base ours --class_pair dog-bird \
    --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 \
    --budget 0.005 $SH --target_idx_file "$IDX"

echo "=== ap8-redundancy.sh finished ==="
