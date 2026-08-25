#!/usr/bin/env bash
#
# appendix.tex, tab:cross-dataset -- the CIFAR-100 rows.
#
# Replaces the TinyImageNet version (appendix/ap2-4.sh and its ap2-1..3 surrogate
# shards), which cost ~74 h of victim training alone: a TinyImageNet ResNet18BN
# victim at 64x64 is 2327 s, against roughly 280 s for the same network on
# CIFAR-100 at 32x32. Same question -- does the selector still help off CIFAR-10 --
# at about a sixth of the compute.
#
# 5 class-pair instances x {GM, SAPA} x {Random, DPP} = 20 runs, one pinned target
# and five victims each, ResNet18BN, poison budget 2e-3.
#
# Why 2e-3: CIFAR-100 has 500 training images per class, exactly like TinyImageNet,
# so the candidate pool is the binding constraint rather than the dataset size.
# 2e-3 of 50,000 is 100 poisons out of 500 candidates -- the same 100 poisons and
# the same one-fifth of the pool the TinyImageNet design used, so the pool argument
# in the paper carries over unchanged.
#
# Why ResNet18BN: the 100-way problem needs a stronger backbone than the ConvNetBN
# used elsewhere in this appendix. Step 0 below trains one clean victim of each so
# the choice is backed by a measured number rather than an assertion.
#
# ~13 h all in (20 surrogates ~2 h, then 20 runs at ~0.55 h). Every unit skips
# itself if already on disk, so a killed allocation just needs this file rerun.
#
#   sh appendix/ap2-cifar100.sh
#   DRY_RUN=1 sh appendix/ap2-cifar100.sh

set -u

# DATASET / MODEL / BUDGET are overridable so the same driver serves every row of
# tab:cross-dataset. Defaults are the CIFAR-100 row; svhn-cells.sh sets SVHN.
DATASET="${DATASET:-CIFAR100}"
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42
NV=5
MODEL="${MODEL:-ResNet18BN}"
BUDGET="${BUDGET:-0.002}"
# SVHN diverges on ~14% of victim seeds at lr 0.1 (a collapsed run sits at the
# 0.196 majority-class rate), so svhn-cells.sh lowers it. Not part of the run
# name, so a directory trained at one lr must be moved aside before changing it.
VICTIM_LR="${VICTIM_LR:-0.1}"
DRY_RUN="${DRY_RUN:-}"
# STEP      probe | surrogates | runs | all   (default all)
# INSTANCES which of the five sampled class pairs to run, e.g. "1 2"
STEP="${STEP:-all}"
INSTANCES="${INSTANCES:-1 2 3 4 5}"

cd /home/mmoslem3/scratch/attack_if
source /home/mmoslem3/ENV/bin/activate

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ap2-cifar100: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

PAIRFILE="target_sets/xdata_pairs_${DATASET}.json"
if [ ! -s "$PAIRFILE" ]; then
    python appendix/pick_pairs.py --dataset "$DATASET" --data_path "$DATA_PATH" \
        --num_pairs 5 --seed "$SEED" --out "$PAIRFILE" >/dev/null || exit 1
fi
PAIRS=$(python -c "
import json;print(' '.join(q['class_pair'] for q in json.load(open('$PAIRFILE'))['pairs']))")
echo "class pairs (<adversarial>-<target>): $PAIRS"
SUBSET=""
for I in $INSTANCES; do
    SUBSET="$SUBSET $(echo $PAIRS | cut -d' ' -f$I)"
done
echo "this shard: instances [$INSTANCES ] ->$SUBSET"

# No --fast_gradmatch here. It was inherited from the TinyImageNet script, where
# ResNet18 at 64x64 with large poison counts needed it. At 32x32 with 100 poisons
# the exact path fits, and it has to be the exact path: tab:cross-dataset puts the
# CIFAR-100 rows next to CIFAR-10 rows crafted with create_graph, and the fast
# branch descends the poison loss rather than the gradient-alignment objective,
# so the two would not be the same attack.
COMMON="--dataset $DATASET --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr $VICTIM_LR --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 1 --num_victims $NV \
    --model $MODEL --budget $BUDGET"

run() {
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
        echo "--- done already: $TAG"; return 0; fi
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then echo "    python final_update.py $* --target_idx_file $IDX"; return 0; fi
    python final_update.py "$@" --target_idx_file "$IDX" || exit 1
}

echo
echo "########## step 0: clean victims, for the backbone justification ##########"
case "$STEP" in probe|all) ;; *) echo "  (STEP=$STEP, skipped)";; esac
if [ "$STEP" = "probe" ] || [ "$STEP" = "all" ]; then
# One clean victim of each backbone under the identical victim schedule, so the
# choice of ResNet18BN over the ConvNetBN used elsewhere rests on a measured
# CIFAR-100 accuracy. ~2 min for ConvNetBN, ~5 min for ResNet18BN, cached after.
FIRST_PAIR=$(echo $PAIRS | cut -d' ' -f1)
for M in ConvNetBN ResNet18BN; do
    echo "--- clean $M on $DATASET"
    if [ -n "$DRY_RUN" ]; then echo "    (skipped in dry run)"; continue; fi
    python final_update.py --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" --model "$M" \
        --class_pair "$FIRST_PAIR" --pair_order poison-target \
        --victim_epochs 50 --victim_lr $VICTIM_LR --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 --num_victims 1 \
        --precompute_only --precompute_part victim 2>&1 | grep -i "victim" | tail -2 \
        || echo "  (probe failed; not fatal)"
done
fi
[ "$STEP" = "probe" ] && { echo "=== ap2-cifar100.sh STEP=probe finished ==="; exit 0; }

echo
if [ "$STEP" = "surrogates" ]; then
    echo "########## surrogates only (20 x $MODEL on $DATASET) ##########"
    SDIR="$CACHE_DIR/surrogates/${DATASET}_${MODEL}_60ep_lr0.1_bs128_seed$SEED"
    FIRST=$(echo $PAIRS | cut -d' ' -f1)
    for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
        if [ -s "$SDIR/net_$i.pt" ]; then echo "--- have surrogate $i"; continue; fi
        echo "--- surrogate $i"
        [ -n "$DRY_RUN" ] && continue
        python final_update.py $COMMON --class_pair "$FIRST" --attack gradmatch \
            --base random --precompute_only --precompute_part surrogate \
            --precompute_id $i || exit 1
    done
    echo "=== ap2-cifar100.sh STEP=surrogates finished ==="; exit 0
fi

echo "########## attack runs ##########"
for PAIR in $SUBSET; do
    IDX="target_sets/xdata_${DATASET}_${MODEL}_${PAIR}.json"
    if [ ! -s "$IDX" ] && [ -z "$DRY_RUN" ]; then
        python appendix/pin_targets.py --dataset "$DATASET" --data_path "$DATA_PATH" \
            --model "$MODEL" --pair "$PAIR" --target_select random \
            --num_targets 1 --num_victims "$NV" --out "$IDX" || exit 1
    fi

    for ATT in gradmatch sapa; do
        case "$ATT" in sapa) SH="--sharp_mode worst --sharp_sigma 0.05"; SN="_worst0.05";; *) SH=""; SN="";; esac

        LABEL="$DATASET | $PAIR | $ATT | Random"
        run "${DATASET}_${MODEL}_${ATT}_random_${PAIR}_b${BUDGET}_eps8_seed${SEED}${SN}_ce5" \
            $COMMON --class_pair "$PAIR" --attack $ATT --base random $SH

        LABEL="$DATASET | $PAIR | $ATT | DPP"
        run "${DATASET}_${MODEL}_${ATT}_ours_${PAIR}_b${BUDGET}_eps8_seed${SEED}_lam1_cosine_seldpp2${SN}_ce5" \
            $COMMON --class_pair "$PAIR" --attack $ATT --base ours --base_dist cosine \
            --lambda_margin 1.0 --sel_dpp --sel_alpha 2.0 $SH
    done
done

echo "=== ap2-cifar100.sh finished ==="
