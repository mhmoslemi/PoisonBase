#!/usr/bin/env bash
#
# ablation.tex tab:alpha-ablation (alpha), shard 5 of 8
#
#     ResNet20BN  b0.01   dpp alpha=4
#     ConvNetBN   b0.005  Greedy (lam=1, no dpp)
#     ConvNetBN   b0.005  dpp alpha=0.5
#
# 5 targets x 4 victims = 20 trials per run, targets pinned to the first 5 of
# target_sets/<model>_gradmatch_dog-bird.json in file order. ~3.3 h.
#
#   sh ab1-5.sh
#   DRY_RUN=1 sh ab1-5.sh

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42
NT="${NUM_TARGETS:-5}"
NV="${NUM_VICTIMS:-4}"
DRY_RUN="${DRY_RUN:-}"

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

if [ -z "$DRY_RUN" ]; then
    python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ab1-5.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }
fi

run() {   # $1 = tag, $2.. = extra flags
    TAG="$1"; shift
    if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
        echo "--- already complete: $TAG"
        return 0
    fi
    echo "=== $M / budget $B / $LABEL ==="
    if [ -n "$DRY_RUN" ]; then
        echo "    python final_update.py --num_surrogates $NSUR $* (tag $TAG)"
        return 0
    fi
    python final_update.py \
        --num_surrogates "$NSUR" "$@" \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model "$M" --attack sapa --base ours \
        --class_pair dog-bird --pair_order poison-target \
        --budget "$B" --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --base_dist cosine \
        --sharp_mode worst --sharp_sigma 0.05 \
        --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets "$NT" --target_select "$TGT" \
        --target_idx_file "target_sets/${M}_gradmatch_dog-bird.json" \
        --num_victims "$NV" --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
}

M=ResNet20BN; B=0.01; TGT=14; NSUR=20
LABEL="dpp alpha=4"
run CIFAR10_ResNet20BN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_seldpp4_worst0.05_ce5_tgt14 \
    --lambda_margin 1.0 --sel_dpp --sel_alpha 4

M=ConvNetBN; B=0.005; TGT=70; NSUR=20
LABEL="Greedy (lam=1, no dpp)"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_worst0.05_ce5_tgt70 \
    --lambda_margin 1.0

M=ConvNetBN; B=0.005; TGT=70; NSUR=20
LABEL="dpp alpha=0.5"
run CIFAR10_ConvNetBN_sapa_ours_dog-bird_b0.005_eps8_seed42_lam1_cosine_seldpp0.5_worst0.05_ce5_tgt70 \
    --lambda_margin 1.0 --sel_dpp --sel_alpha 0.5

echo "=== ab1-5.sh finished ==="
