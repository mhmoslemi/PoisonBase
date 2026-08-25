#!/usr/bin/env bash
#
# appendix.tex, tab:broad-cifar
#
# tab:broad-cifar -- six additional ordered CIFAR-10 class pairs plus the main
# pair as a reference row, ConvNetBN, eps=5e-3, FC / GM / SAPA, Random vs DPP.
# 42 runs.
#
# Pairs, as the paper writes them (target -> adversarial):
#   cat->dog  deer->horse  automobile->truck  bird->airplane  ship->frog  truck->cat
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
#   sh appendix/ap1-broad.sh
#   DRY_RUN=1 sh appendix/ap1-broad.sh

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
        echo "ap1-broad.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
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

# "<adv>-<target>" = the paper's "target -> adv" reversed
# six new pairs, plus the main pair as the reference row of tab:broad-cifar
PAIRS="${PAIRS:-dog-cat horse-deer truck-automobile airplane-bird frog-ship cat-truck dog-bird}"

for PAIR in $PAIRS; do
    IDX="target_sets/appx_broad_ConvNetBN_${PAIR}.json"
    pin "$PAIR" "$IDX" ConvNetBN
    for ATT in fc gradmatch sapa; do
        case "$ATT" in sapa) SHARP="--sharp_mode worst --sharp_sigma 0.05"; SN="_worst0.05";; *) SHARP=""; SN="";; esac
        LABEL="broad | $PAIR | $ATT | Random"
        run "CIFAR10_ConvNetBN_${ATT}_random_${PAIR}_b0.005_eps8_seed42${SN}_ce5" \
            $COMMON --model ConvNetBN --attack $ATT --base random \
            --class_pair "$PAIR" --budget 0.005 $SHARP --target_idx_file "$IDX"
        LABEL="broad | $PAIR | $ATT | DPP"
        run "CIFAR10_ConvNetBN_${ATT}_ours_${PAIR}_b0.005_eps8_seed42_lam1_cosine_seldpp2${SN}_ce5" \
            $COMMON --model ConvNetBN --attack $ATT --base ours \
            --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 \
            --class_pair "$PAIR" --budget 0.005 $SHARP --target_idx_file "$IDX"
    done
done

echo "=== ap1-broad.sh finished ==="
