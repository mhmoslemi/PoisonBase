#!/usr/bin/env bash
#
# cross_arch.tex -- the four Random cells whose attack/victim architecture is VGG13BN:
#
#     dog--bird & FC & ConvNet & VGG13 & <Random>
#     dog--bird & FC & ResNet20BN & VGG13 & <Random>
#     dog--bird & GM & ConvNet & VGG13 & <Random>
#     dog--bird & GM & ResNet20BN & VGG13 & <Random>
#
# 5 pinned targets x victims 0-3 = 20 trials per cell, same as the DPP column.
#
#   sh xr3.sh

set -u

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "xr3.sh: no CUDA device visible -- get a GPU allocation first"; exit 1; }

TAG=CIFAR10_VGG13BN_fc_random_dog-bird_b0.005_eps8_seed42_selarchConvNetBN_ce5_tgt3
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== S=ConvNetBN -> A=VGG13BN | fc | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
else
    echo "=== cross-arch random | S=ConvNetBN -> A=V=VGG13BN | fc / dog-bird / budget 0.005 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model VGG13BN --sel_model ConvNetBN --attack fc --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select 3 \
        --target_idx_file "target_sets/xarch_VGG13BN_fc_dog-bird_b0.005.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
fi

TAG=CIFAR10_VGG13BN_fc_random_dog-bird_b0.005_eps8_seed42_selarchResNet20BN_ce5_tgt3
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== S=ResNet20BN -> A=VGG13BN | fc | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
else
    echo "=== cross-arch random | S=ResNet20BN -> A=V=VGG13BN | fc / dog-bird / budget 0.005 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model VGG13BN --sel_model ResNet20BN --attack fc --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select 3 \
        --target_idx_file "target_sets/xarch_VGG13BN_fc_dog-bird_b0.005.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
fi

TAG=CIFAR10_VGG13BN_gradmatch_random_dog-bird_b0.005_eps8_seed42_selarchConvNetBN_ce5_tgt50
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== S=ConvNetBN -> A=VGG13BN | gradmatch | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
else
    echo "=== cross-arch random | S=ConvNetBN -> A=V=VGG13BN | gradmatch / dog-bird / budget 0.005 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model VGG13BN --sel_model ConvNetBN --attack gradmatch --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
    --craft_lowmem --craft_batch 256 --fast_gradmatch \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select 50 \
        --target_idx_file "target_sets/xarch_VGG13BN_gradmatch_dog-bird_b0.005.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
fi

TAG=CIFAR10_VGG13BN_gradmatch_random_dog-bird_b0.005_eps8_seed42_selarchResNet20BN_ce5_tgt50
if grep -q "==== $TAG : ASR = " "$OUT_DIR/$TAG/log.txt" 2>/dev/null; then
    echo "=== S=ResNet20BN -> A=VGG13BN | gradmatch | already complete:"
    grep -o "ASR = .*====" "$OUT_DIR/$TAG/log.txt" | tail -1
else
    echo "=== cross-arch random | S=ResNet20BN -> A=V=VGG13BN | gradmatch / dog-bird / budget 0.005 ==="
    python final_update.py \
        --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
        --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
        --model VGG13BN --sel_model ResNet20BN --attack gradmatch --base random \
        --class_pair dog-bird --pair_order poison-target \
        --budget 0.005 --epsilon 0.0313725 \
        --craft_steps 250 --craft_alpha 0.0039216 \
        --restarts 8 --craft_ensemble 5 \
    --craft_lowmem --craft_batch 256 --fast_gradmatch \
        --num_surrogates 20 --surrogate_epochs 60 --surrogate_decay 35 45 \
        --num_targets 5 --target_select 50 \
        --target_idx_file "target_sets/xarch_VGG13BN_gradmatch_dog-bird_b0.005.json" \
        --num_victims 4 --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
        --victim_decay 40 --victim_wd 0.0 \
        --clean_baseline || exit 1
fi

echo "=== xr3.sh finished ==="
