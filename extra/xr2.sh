#!/usr/bin/env bash
#
# cross_arch.tex -- the four Random cells whose attack/victim architecture is ResNet20BN:
#
#     dog--bird & FC & ConvNet & ResNet20BN & <Random>
#     dog--bird & FC & VGG13 & ResNet20BN & <Random>
#     dog--bird & GM & ConvNet & ResNet20BN & <Random>
#     dog--bird & GM & VGG13 & ResNet20BN & <Random>
#
# 5 pinned targets x victims 0-3 = 20 trials per cell, same as the DPP column.
#
#   sh xr2.sh

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "xr2.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_ResNet20BN_fc_random_dog-bird_b0.005_eps8_seed42_selarchConvNetBN_ce5_tgt10
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== S=ConvNetBN -> A=ResNet20BN | fc | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
else
    echo "=== cross-arch random | S=ConvNetBN -> A=V=ResNet20BN | fc / dog-bird / budget 0.005 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model ResNet20BN --sel_model ConvNetBN --attack fc --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select 10 \
        --target_idx_file "target_sets/xarch_ResNet20BN_fc_dog-bird_b0.005.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
fi

TAG=CIFAR10_ResNet20BN_fc_random_dog-bird_b0.005_eps8_seed42_selarchVGG13BN_ce5_tgt10
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== S=VGG13BN -> A=ResNet20BN | fc | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
else
    echo "=== cross-arch random | S=VGG13BN -> A=V=ResNet20BN | fc / dog-bird / budget 0.005 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model ResNet20BN --sel_model VGG13BN --attack fc --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select 10 \
        --target_idx_file "target_sets/xarch_ResNet20BN_fc_dog-bird_b0.005.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
fi

TAG=CIFAR10_ResNet20BN_gradmatch_random_dog-bird_b0.005_eps8_seed42_selarchConvNetBN_ce5_tgt14
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== S=ConvNetBN -> A=ResNet20BN | gradmatch | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
else
    echo "=== cross-arch random | S=ConvNetBN -> A=V=ResNet20BN | gradmatch / dog-bird / budget 0.005 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model ResNet20BN --sel_model ConvNetBN --attack gradmatch --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select 14 \
        --target_idx_file "target_sets/xarch_ResNet20BN_gradmatch_dog-bird_b0.005.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
fi

TAG=CIFAR10_ResNet20BN_gradmatch_random_dog-bird_b0.005_eps8_seed42_selarchVGG13BN_ce5_tgt14
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== S=VGG13BN -> A=ResNet20BN | gradmatch | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
else
    echo "=== cross-arch random | S=VGG13BN -> A=V=ResNet20BN | gradmatch / dog-bird / budget 0.005 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model ResNet20BN --sel_model VGG13BN --attack gradmatch --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select 14 \
        --target_idx_file "target_sets/xarch_ResNet20BN_gradmatch_dog-bird_b0.005.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
fi

echo "=== xr2.sh finished ==="
