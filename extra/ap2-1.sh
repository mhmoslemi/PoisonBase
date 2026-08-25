#!/usr/bin/env bash
#
# appendix.tex, tab:cross-dataset -- chunk 1 of 4.
#
# the CIFAR-10 rows still missing, the 4 remaining clean victims, and surrogates 0-5
#
# ~8.2 h, from measured rates: a ResNet18BN TinyImageNet clean victim is 2327 s,
# a surrogate 2792 s (60 epochs vs the victim's 50), a craft at N_p=100 is 1666 s,
# so one attack run (1 craft + 5 victims) is 3.69 h.
#
# Every unit is skipped if already on disk, so a killed job just needs this same
# file rerun. Needs --mem=32G.
#
#   sh appendix/ap2-1.sh

set -u

DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42
NV=5
TINY_MODEL=ResNet18BN
DRY_RUN="${DRY_RUN:-}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ap2-1.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi
[ -s "$DATA_PATH/tinyimagenet.pt" ] || {
    echo "TinyImageNet not prepared -- python appendix/prep_tinyimagenet.py --src tiny-imagenet-200.zip"; exit 1; }

run() {
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
        echo "--- done already: $TAG"; return 0; fi
    echo "=== $LABEL ==="
    if [ -n "$DRY_RUN" ]; then echo "    python final_update.py $*"; return 0; fi
    python final_update.py "$@" || exit 1
}

TINY_BASE="--dataset TinyImageNet --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR \
    --model $TINY_MODEL --attack gradmatch --base random \
    --class_pair n01443537-n01629819 --pair_order poison-target \
    --budget 0.001 --epsilon 0.0313725 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --num_victims $NV --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0"
SDIR="$CACHE_DIR/surrogates/TinyImageNet_${TINY_MODEL}_60ep_lr0.1_bs128_seed$SEED"

echo "########## CIFAR-10 reference rows ##########"
CIDX="target_sets/appx_matched_dog-bird.json"
[ -s "$CIDX" ] || [ -n "$DRY_RUN" ] || python appendix/pin_targets.py --model ConvNetBN \
    --pair dog-bird --target_select random --num_targets 5 --num_victims $NV --out "$CIDX" || exit 1
CIFAR_BASE="--dataset CIFAR10 --data_path $DATA_PATH --seed $SEED \
    --cache_dir $CACHE_DIR --out_dir $OUT_DIR --pair_order poison-target \
    --epsilon 0.0313725 --craft_steps 250 --craft_alpha 0.0039216 \
    --restarts 8 --craft_ensemble 5 \
    --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
    --victim_decay 40 --victim_wd 0.0 --clean_baseline \
    --target_select random --num_targets 5 --num_victims $NV \
    --model ConvNetBN --class_pair dog-bird --budget 0.005"
for ATT in gradmatch sapa; do
    case "$ATT" in sapa) SH_="--sharp_mode worst --sharp_sigma 0.05"; SN="_worst0.05";; *) SH_=""; SN="";; esac
    LABEL="CIFAR-10 | $ATT | Random"
    run "CIFAR10_ConvNetBN_${ATT}_random_dog-bird_b0.005_eps8_seed42${SN}_ce5" \
        $CIFAR_BASE --attack $ATT --base random $SH_ --target_idx_file "$CIDX"
    LABEL="CIFAR-10 | $ATT | DPP"
    run "CIFAR10_ConvNetBN_${ATT}_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp2${SN}_ce5" \
        $CIFAR_BASE --attack $ATT --base ours --base_dist cosine --lambda_margin 1.0 \
        --sel_dpp --sel_alpha 2.0 $SH_ --target_idx_file "$CIDX"
done

echo
echo "########## TinyImageNet clean victims (ResNet18BN) ##########"
VDIR="$CACHE_DIR/clean_victims/TinyImageNet_${TINY_MODEL}_50ep_lr0.1_bs125_wd0_seed$SEED"
echo "  have $(ls "$VDIR" 2>/dev/null | wc -l)/$NV -- ~39 min each, no output until one lands"
if [ "$(ls "$VDIR" 2>/dev/null | wc -l)" -lt "$NV" ] && [ -z "$DRY_RUN" ]; then
    python final_update.py $TINY_BASE --precompute_only --precompute_part victim || exit 1
fi

echo
echo "########## TinyImageNet surrogates 0,1,2,3,4,5 ##########"
for i in 0 1 2 3 4 5; do
    if [ -s "$SDIR/net_$i.pt" ]; then echo "--- have surrogate $i"; continue; fi
    echo "--- surrogate $i (~47 min)"
    [ -n "$DRY_RUN" ] && continue
    python final_update.py $TINY_BASE --precompute_only --precompute_part surrogate \
        --precompute_id "$i" || exit 1
done

echo "=== ap2-1.sh finished ==="
