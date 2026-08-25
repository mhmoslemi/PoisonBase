#!/usr/bin/env bash
#
# appendix.tex, tab:augmentation-aware -- unmatched vs matched craft-time augmentation
#
# ConvNetBN / bird -> dog (--class_pair dog-bird), GM and SAPA at eps=5e-3,
# victim augmentation Crop+Flip or RandAugment, Random vs DPP.
#
#   Unmatched  poisons optimized on unaugmented images (--no_craft_aug), the victim
#              adds augmentation afterwards.  Note the main-sweep crafts do NOT
#              qualify: craft_aug defaults to on, so they were optimized through the
#              full DSA strategy.  These are new crafts.
#   Matched    the victim's own augmentation is sampled during crafting.  Crop+Flip
#              maps onto --dsa_strategy crop_flip.  RandAugment has no gradient, so a
#              matched attacker is undefined for it -- appendix.tex marks those two
#              entries n/a and this script does not run them.
#
# Craft-time augmentation is part of the run name (_craftnoaug / _craftcropflip), so
# the two conditions cannot overwrite each other, and aug.sh is told which of them to
# replay through CRAFT_AUG / DSA_STRATEGY.
#
# These crafts follow the APPENDIX protocol -- targets sampled uniformly, so the run
# name has no _tgt<N> suffix. aug.sh defaults to the main sweep's difficulty-pinned
# naming, which is why the replays must pass TARGET_SELECT=random; without it aug.sh
# reads sweep_config.json, builds a _tgt<N> name that no craft ever wrote, and skips
# every combo with "no paired poisons" (and skips sapa/dog-bird even earlier, since
# that combo has no difficulty entry at all).
#
# STEP selects the phase, so no single invocation runs longer than ~8 h:
#   STEP=craft    8 crafts                                    ~5.5 h
#   STEP=cf       replay all 8 under Crop+Flip                ~2.1 h
#   STEP=ra_gm    replay the GM unmatched pair under RandAug   ~5.5 h
#   STEP=ra_sapa  replay the SAPA unmatched pair under RandAug ~5.5 h
#
#   STEP=craft sh appendix/ap4-augaware.sh
#   DRY_RUN=1 STEP=cf sh appendix/ap4-augaware.sh

set -u

DATA_PATH=/home/mmoslem3/scratch/data
SEED=42
NT="${NUM_TARGETS:-5}"
NV="${NUM_VICTIMS:-5}"
STEP="${STEP:-craft}"
# Which selections to replay. Both by default; the appendix/final shards set one
# at a time so a single RandAugment run (25 trials x 645 s) fits one allocation.
# PAIR_SELS stays "random dpp" regardless -- the pinned target set is the
# intersection over both, so it must not change when only one is replayed.
SELS="${SELS:-random dpp}"
DRY_RUN="${DRY_RUN:-}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ap4: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

IDX="target_sets/appx_matched_dog-bird.json"
if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --model ConvNetBN --pair dog-bird \
        --target_select random --num_targets "$NT" --num_victims "$NV" --out "$IDX" || exit 1
fi

COMMON="--dataset CIFAR10 --data_path $DATA_PATH --seed $SEED \
    --cache_dir ./cache --out_dir ours_result --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets $NT --num_victims $NV \
    --model ConvNetBN --class_pair dog-bird --budget 0.005"

craft() {   # $1 = tag, $2.. = flags
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "ours_result/$TAG/log.txt" 2>/dev/null; then
        echo "--- already complete: $TAG"; return 0; fi
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then echo "    python final_update.py $*"; return 0; fi
    python final_update.py "$@" --target_idx_file "$IDX" || exit 1
}

if [ "$STEP" = "craft" ]; then
    for ATT in gradmatch sapa; do
        case "$ATT" in sapa) SH="--sharp_mode worst --sharp_sigma 0.05"; SN="_worst0.05";; *) SH=""; SN="";; esac
        for COND in unmatched matched; do
            case "$COND" in
                unmatched) AUGF="--no_craft_aug";           CN="_craftnoaug";;
                matched)   AUGF="--dsa_strategy crop_flip"; CN="_craftcropflip";;
            esac
            LABEL="craft | $ATT | $COND | Random"
            craft "CIFAR10_ConvNetBN_${ATT}_random_dog-bird_b0.005_eps8_seed42${SN}${CN}_ce5" \
                $COMMON --attack $ATT --base random $SH $AUGF
            LABEL="craft | $ATT | $COND | DPP"
            craft "CIFAR10_ConvNetBN_${ATT}_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2${SN}${CN}_ce5" \
                $COMMON --attack $ATT --base ours --base_dist cosine --lambda_margin 1.0 \
                --sel_dpp --sel_alpha 2.0 $SH $AUGF
        done
    done
    echo "=== ap4 STEP=craft finished ==="; exit 0
fi

# ---- replays. CRAFT_AUG / DSA_STRATEGY tell aug.sh which crafts to read ---------
replay() {   # $1 = attack, $2 = augs, $3 = craft condition
    case "$3" in
        unmatched) CA="--no_craft_aug"; DS="";;
        matched)   CA="";               DS="crop_flip";;
    esac
    echo "=== replay | $1 | augs=$2 | craft=$3 ==="
    [ -n "$DRY_RUN" ] && { echo "    aug.sh CRAFT_AUG='$CA' DSA_STRATEGY='$DS'"; return 0; }
    env MODEL=ConvNetBN ATTACK="$1" CLASS_PAIR="dog-bird" \
        BUDGETS="0.005" SELS="random dpp" PAIR_SELS="random dpp" \
        AUGS="$2" NUM_TARGETS="$NT" NUM_VICTIMS="$NV" SELS="$SELS" \
        TARGET_SELECT=random \
        CRAFT_AUG="$CA" DSA_STRATEGY="$DS" sh ./aug.sh || exit 1
}

case "$STEP" in
    cf)       for A in gradmatch sapa; do for C in unmatched matched; do replay $A standard $C; done; done;;
    ra_gm)    replay gradmatch randaug unmatched;;
    ra_sapa)  replay sapa randaug unmatched;;
    *) echo "unknown STEP='$STEP' (craft | cf | ra_gm | ra_sapa)"; exit 1;;
esac

echo "=== ap4 STEP=$STEP finished ==="
