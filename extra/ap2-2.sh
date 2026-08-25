#!/usr/bin/env bash
#
# appendix.tex, tab:cross-dataset -- chunk 2 of 4.
#
# surrogates 6-15
#
# ~7.8 h, from measured rates: a ResNet18BN TinyImageNet clean victim is 2327 s,
# a surrogate 2792 s (60 epochs vs the victim's 50), a craft at N_p=100 is 1666 s,
# so one attack run (1 craft + 5 victims) is 3.69 h.
#
# Every unit is skipped if already on disk, so a killed job just needs this same
# file rerun. Needs --mem=32G.
#
#   sh appendix/ap2-2.sh

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
        echo "ap2-2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
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

echo
echo "########## TinyImageNet surrogates 6,7,8,9,10,11,12,13,14,15 ##########"
for i in 6 7 8 9 10 11 12 13 14 15; do
    if [ -s "$SDIR/net_$i.pt" ]; then echo "--- have surrogate $i"; continue; fi
    echo "--- surrogate $i (~47 min)"
    [ -n "$DRY_RUN" ] && continue
    python final_update.py $TINY_BASE --precompute_only --precompute_part surrogate \
        --precompute_id "$i" || exit 1
done

echo "=== ap2-2.sh finished ==="
