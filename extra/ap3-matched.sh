#!/usr/bin/env bash
#
# appendix.tex, tab:matched-architecture
#
# tab:matched-architecture -- one target set, three architectures.
#
# Five dog-class targets with bird as the adversarial class (so --class_pair
# dog-bird), GM and SAPA at eps=5e-3, Random vs DPP, ConvNetBN / ResNet20BN /
# VGG13BN.  12 runs.
#
# The whole point is that the five targets are identical everywhere, so the file
# is pinned once with ConvNetBN's clean victims and reused for all three models.
# Selection still uses each architecture's own surrogates, as the text requires.
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
#   sh appendix/ap3-matched.sh
#   DRY_RUN=1 sh appendix/ap3-matched.sh

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
        echo "ap3-matched.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
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

for M in ${MODELS:-ConvNetBN ResNet20BN VGG13BN}; do
    case "$M" in VGG13BN) MEM="--craft_lowmem --craft_batch 256 --fast_gradmatch";; *) MEM="";; esac
    for ATT in gradmatch sapa; do
        case "$ATT" in sapa) SHARP="--sharp_mode worst --sharp_sigma 0.05"; SN="_worst0.05";; *) SHARP=""; SN="";; esac
        LABEL="matched | $M | $ATT | Random"
        run "CIFAR10_${M}_${ATT}_random_dog-bird_b0.005_eps8_seed42${SN}_ce5" \
            $COMMON --model $M --attack $ATT --base random --class_pair dog-bird \
            --budget 0.005 $SHARP $MEM --target_idx_file "$IDX"
        LABEL="matched | $M | $ATT | DPP"
        run "CIFAR10_${M}_${ATT}_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2${SN}_ce5" \
            $COMMON --model $M --attack $ATT --base ours --class_pair dog-bird \
            --base_dist cosine --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 \
            --budget 0.005 $SHARP $MEM --target_idx_file "$IDX"
    done
done

echo "=== ap3-matched.sh finished ==="
