#!/usr/bin/env bash
#
# appendix.tex, tab:cross-dataset -- the WHOLE table, start to finish.
#
# Self-contained: it trains whatever is missing, pins its own targets, and runs
# every cell. Nothing else has to be run first and nothing depends on it.
#
#   CIFAR-10 rows      ConvNetBN, bird -> dog, eps=5e-3, GM and SAPA, Random/DPP
#   TinyImageNet rows  ResNet18BN, 5 instances x 1 target, eps=1e-3, GM and SAPA,
#                      Random/DPP
#
# ResNet18 on TinyImageNet because ConvNetBN only reaches ~24% clean accuracy on
# 200 classes. eps=1e-3 there because a class holds 500 images and 5e-3 would need
# 500 poisons -- the pool would be consumed and every selector would return the
# same set.
#
# WHAT IT DOES, in order:
#   1. the 4 CIFAR-10 runs                                        ~2.7 h
#   2. 5 TinyImageNet clean victims                               ~2-5 h
#   3. 20 TinyImageNet surrogates, one at a time                  ~11-22 h
#   4. 5 target sets pinned, then 20 TinyImageNet attack runs     ~51-103 h
#
# So 65-130 h in total. It is fully resumable: every net and every trial is written
# as it completes, a finished run is skipped in a second, and a killed run picks up
# at its first missing trial. Just run it again after each wall-clock kill until it
# prints "finished".
#
# Needs --mem=32G (TinyImageNet is 4.6 GB as float32) and tinyimagenet.pt, which
# appendix/prep_tinyimagenet.py builds.
#
# Split into two chunks so the first fits one allocation:
#   sh appendix/ap2-b1.sh    # MAX_H=8, stops cleanly at the 8 h mark
#   sh appendix/ap2-b2.sh    # everything still missing, no cap, rerun until done
#
#   sh appendix/ap2-b.sh              # or the whole thing in one go
#   MAX_H=6 sh appendix/ap2-b.sh      # any budget you like
#   DRY_RUN=1 sh appendix/ap2-b.sh    # print what it would do and stop

set -u

DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42
NT="${NUM_TARGETS:-5}"
NV="${NUM_VICTIMS:-5}"
TINY_MODEL="${TINY_MODEL:-ResNet18BN}"
DRY_RUN="${DRY_RUN:-}"
# MAX_H caps how long this invocation keeps STARTING work. 0 = no cap. It never
# interrupts a unit in flight: before each one it checks that the time left is at
# least as long as the previous unit took, so nothing is killed halfway.
MAX_H="${MAX_H:-0}"
_T0=$(date +%s)
_LAST=0

have_time() {
    [ "$MAX_H" = "0" ] && return 0
    _now=$(date +%s)
    _left=$(( MAX_H * 3600 - (_now - _T0) ))
    if [ "$_left" -le "$_LAST" ]; then
        echo
        echo "=== MAX_H=$MAX_H reached (${_left}s left, last unit took ${_LAST}s)."
        echo "=== Stopping cleanly. Everything finished is on disk; rerun to continue."
        exit 0
    fi
    _UNIT_START=$_now
    return 0
}

mark_unit() { _LAST=$(( $(date +%s) - _UNIT_START )); }

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ap2-b: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

if [ ! -s "$DATA_PATH/tinyimagenet.pt" ]; then
    echo "TinyImageNet is not prepared: $DATA_PATH/tinyimagenet.pt is missing."
    echo "Build it once (login node, compute nodes have no internet):"
    echo "    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    echo "    python appendix/prep_tinyimagenet.py --src tiny-imagenet-200.zip"
    exit 1
fi

run() {   # $1 = run dir name, $2.. = flags
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
        echo "--- done already: $TAG"
        return 0
    fi
    have_time
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then echo "    python final_update.py $*"; return 0; fi
    python final_update.py "$@" || exit 1
    mark_unit
}

# ---------------------------------------------------------------- 1. CIFAR-10 --
echo "########## 1/4  CIFAR-10 reference rows (ConvNetBN) ##########"
CIDX="target_sets/appx_matched_dog-bird.json"
if [ ! -s "$CIDX" ] && [ -z "$DRY_RUN" ]; then
    python appendix/pin_targets.py --model ConvNetBN --pair dog-bird \
        --target_select random --num_targets "$NT" --num_victims "$NV" --out "$CIDX" || exit 1
fi

CIFAR="--dataset CIFAR10 --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets $NT --num_victims $NV \
    --model ConvNetBN --class_pair dog-bird --budget 0.005"

for ATT in gradmatch sapa; do
    case "$ATT" in sapa) SH_="--sharp_mode worst --sharp_sigma 0.05"; SN="_worst0.05";; *) SH_=""; SN="";; esac
    LABEL="CIFAR-10 | $ATT | Random"
    run "CIFAR10_ConvNetBN_${ATT}_random_dog-bird_b0.005_eps8_seed42${SN}_ce5" \
        $CIFAR --attack $ATT --base random $SH_ --target_idx_file "$CIDX"
    LABEL="CIFAR-10 | $ATT | DPP"
    run "CIFAR10_ConvNetBN_${ATT}_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2${SN}_ce5" \
        $CIFAR --attack $ATT --base ours --base_dist cosine --lambda_margin 1.0 \
        --sel_dpp --sel_alpha 2.0 $SH_ --target_idx_file "$CIDX"
done

# ------------------------------------------------- 2 + 3. TinyImageNet pools --
TINY_BASE="--dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR \
    --model $TINY_MODEL --attack gradmatch --base random \
    --class_pair n01443537-n01629819 --pair_order poison-target \
    --budget 0.001 --epsilon 0.0313725 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_victims $NV --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0"

VDIR="$CACHE_DIR/clean_victims/TinyImageNet_${TINY_MODEL}_50ep_lr0.1_bs125_wd0_seed$SEED"
SDIR="$CACHE_DIR/surrogates/TinyImageNet_${TINY_MODEL}_60ep_lr0.1_bs128_seed$SEED"

echo
echo "########## 2/4  TinyImageNet clean victims ($TINY_MODEL) ##########"
echo "  have $(ls "$VDIR" 2>/dev/null | wc -l)/$NV -- no output until one finishes"
if [ "$(ls "$VDIR" 2>/dev/null | wc -l)" -lt "$NV" ]; then
    if [ -n "$DRY_RUN" ]; then echo "    precompute victims"; else
        have_time
        python final_update.py $TINY_BASE --precompute_only --precompute_part victim || exit 1
        mark_unit
    fi
fi

echo
echo "########## 3/4  TinyImageNet surrogates ($TINY_MODEL, K=20) ##########"
echo "  have $(ls "$SDIR" 2>/dev/null | wc -l)/20 -- trained one at a time, each is saved as it lands"
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
    if [ -s "$SDIR/net_$i.pt" ]; then continue; fi
    if [ -n "$DRY_RUN" ]; then echo "--- surrogate $i"; continue; fi
    have_time
    echo "--- surrogate $i"
    python final_update.py $TINY_BASE --precompute_only --precompute_part surrogate \
        --precompute_id "$i" || exit 1
    mark_unit
done

# ------------------------------------------------------- 4. TinyImageNet runs --
echo
echo "########## 4/4  TinyImageNet attack runs ##########"
TINY_PAIRS="n01443537-n01629819 n01641577-n01644900 n01698640-n01742172 n01768244-n01770081 n01774384-n01774750"

for PAIR in $TINY_PAIRS; do
    IDX="target_sets/appx_tiny_${TINY_MODEL}_${PAIR}.json"
    if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
        python appendix/pin_targets.py --dataset TinyImageNet --model "$TINY_MODEL" \
            --pair "$PAIR" --target_select random --num_targets 1 --num_victims "$NV" \
            --out "$IDX" || exit 1
    fi
    TINY="--dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
        --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
        --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 --craft_lowmem --craft_batch 256 --fast_gradmatch \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 --clean_baseline \
        --target_select random --num_targets 1 --num_victims $NV \
        --model $TINY_MODEL --class_pair $PAIR --budget 0.001"
    for ATT in gradmatch sapa; do
        case "$ATT" in sapa) SH_="--sharp_mode worst --sharp_sigma 0.05"; SN="_worst0.05";; *) SH_=""; SN="";; esac
        LABEL="TinyImageNet | $PAIR | $ATT | Random"
        run "TinyImageNet_${TINY_MODEL}_${ATT}_random_${PAIR}_b0.001_eps8_seed42${SN}_ce5" \
            $TINY --attack $ATT --base random $SH_ --target_idx_file "$IDX"
        LABEL="TinyImageNet | $PAIR | $ATT | DPP"
        run "TinyImageNet_${TINY_MODEL}_${ATT}_ours_${PAIR}_b0.001_eps8_seed42_lam1_cosine_seldpp2${SN}_ce5" \
            $TINY --attack $ATT --base ours --base_dist cosine --lambda_margin 1.0 \
            --sel_dpp --sel_alpha 2.0 $SH_ --target_idx_file "$IDX"
    done
done

echo
echo "=== ap2-b.sh finished: the whole cross-dataset table ==="
