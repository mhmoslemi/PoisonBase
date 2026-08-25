#!/usr/bin/env bash
#
# appendix.tex, tab:utility-defense -- EPIC and FRIENDS at matched clean utility
#
# ConvNetBN, eps=2e-2, GM and SAPA, dog -> bird (--class_pair dog-bird) and
# frog -> airplane (--class_pair frog-airplane), Random vs DPP.  16 poisoned runs.
#
# The table's rule is that each defense is retuned until its CLEAN accuracy is
# within two points of the undefended ConvNetBN, using clean data only, and then
# frozen. The reference is the undefended ConvNetBN under THIS protocol, not the
# ResNet20BN numbers of tab:defense-robustness -- the two tables are not directly
# comparable, and the appendix text now says so.
#
# Strength ordering: for EPIC a SMALLER --epic_subset_size keeps less of the
# training set each round and is the stronger setting; for FRIENDS a LARGER
# --noise_eps is stronger. Stage 1 prints both sweeps weakest-first.
#
# That is a search, so this runs in two stages:
#
#   STAGE=1  train clean ConvNetBN under a range of defense strengths and print the
#            clean CTA of each. No poisons, no attack outcomes -- exactly what the
#            text requires for calibration.
#   STAGE=2  you pass the chosen strengths back in and the 16 poisoned runs go out.
#
# Two stages on purpose: "within two percentage points" is a judgement call, and
# hard-coding a guess would quietly make it for you. Undefended ConvNetBN sits at
# ~80.5, so the band is roughly 78.5-80.5.
#
#   STAGE=1 sh appendix/ap5-utilmatch.sh
#   STAGE=2 EPIC_KEEP=<s> NOISE_EPS_SET=<e> sh appendix/ap5-utilmatch.sh

set -u

STAGE="${STAGE:-1}"
EPIC_KEEP="${EPIC_KEEP:-}"
NOISE_EPS_SET="${NOISE_EPS_SET:-}"
NV="${NUM_VICTIMS:-5}"
NT="${NUM_TARGETS:-5}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "ap5: no CUDA device visible -- get a GPU allocation first"; exit 1; }

if [ "$STAGE" = "1" ]; then
    echo "=== stage 1: clean-utility calibration (no poisons) ==="
    for S in 0.5 0.3 0.2 0.1 0.05; do   # weakest -> strongest
        echo "--- EPIC subset_size=$S ---"
        MODEL=ConvNetBN ATTACK=gradmatch CLASS_PAIR="dog-bird" BUDGETS="0.02" \
            SELS="random" DEFENSES="epic" NUM_VICTIMS="$NV" NUM_TARGETS=1 \
            EPIC_SUBSET="$S" sh ./defense.sh 2>&1 | grep -i "defended clean" || true
    done
    for E in 2 4 6 8 12; do              # weakest -> strongest
        echo "--- FRIENDS noise_eps=$E ---"
        MODEL=ConvNetBN ATTACK=gradmatch CLASS_PAIR="dog-bird" BUDGETS="0.02" \
            SELS="random" DEFENSES="friends" NUM_VICTIMS="$NV" NUM_TARGETS=1 \
            NOISE_EPS="$E" sh ./defense.sh 2>&1 | grep -i "defended clean" || true
    done
    echo
    echo "pick the STRONGEST setting still within 2 points of the undefended ~80.5, then:"
    echo "   STAGE=2 EPIC_KEEP=<s> NOISE_EPS_SET=<e> sh appendix/ap5-utilmatch.sh"
    exit 0
fi

[ -n "$EPIC_KEEP" ] && [ -n "$NOISE_EPS_SET" ] || {
    echo "stage 2 needs EPIC_KEEP and NOISE_EPS_SET -- run stage 1 first"; exit 1; }

echo "=== stage 2: EPIC subset=$EPIC_KEEP, FRIENDS eps=$NOISE_EPS_SET ==="
for PAIR in ${PAIRS:-dog-bird frog-airplane}; do
    for ATT in gradmatch sapa; do
        for DEF in ${DEFS:-epic friends}; do
            case "$DEF" in
                epic)    KNOB="EPIC_SUBSET=$EPIC_KEEP";;
                friends) KNOB="NOISE_EPS=$NOISE_EPS_SET";;
                *) echo "unknown defense '$DEF'"; exit 1;;
            esac
            echo "--- $PAIR | $ATT | $DEF ---"
            env MODEL=ConvNetBN ATTACK="$ATT" CLASS_PAIR="$PAIR" BUDGETS="0.02" \
                SELS="random dpp" DEFENSES="$DEF" NUM_VICTIMS="$NV" NUM_TARGETS="$NT" \
                $KNOB sh ./defense.sh || exit 1
        done
    done
done

echo "=== ap5-utilmatch.sh finished ==="
