#!/usr/bin/env bash

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

# MODELS=(ConvNetBN VGG13BN ResNet20BN)
# ATTACKS=(fc gradmatch)
# BASES=(random ours)
# CLASS_PAIRS=(dog-bird frog-airplane)
MODEL=ResNet20BN
ATTACK=fc
BASE=random
CLASS_PAIR=frog-airplane

PAIR_ORDER=poison-target

EPSILON=0.0313725         # 16/255. use 0.0313725 for 8/255
CRAFT_STEPS=250
CRAFT_ALPHA=0.0039216   # 1/255. fc: PGD sign step. gradmatch: signed-Adam lr
RESTARTS=8
CRAFT_ENSEMBLE=5        # 0 = use all surrogates

# easiest | hardest | random | first, or a difficulty degree 0..100
# (0 == easiest, 100 == hardest). Numeric degrees get a _tgt<N> run-name suffix.
TARGET_SELECT=2
# NUM_TARGETS=10
NUM_TARGETS=4

BASE_DIST=l2
LAMBDA=1.0

NUM_SURROGATES=5
SURROGATE_EPOCHS=60
SURROGATE_DECAY="35 45"


# NUM_VICTIMS=6
NUM_VICTIMS=3
# NUM_VICTIMS=5
VICTIM_EPOCHS=50
VICTIM_LR=0.1
VICTIM_BS=125
VICTIM_DECAY="40"

# VGG13BN + gradmatch at large poison counts will not fit the second-order graph.
# Set FAST="--fast_gradmatch" there.
FAST=""

# Fits the same (exact) objective in memory by doing one surrogate and one
# CRAFT_BATCH slice of poisons at a time.  Leave empty for small budgets: the run
# is then bit-for-bit the old code path.  Set LOWMEM="--craft_lowmem" for the
# budgets that OOM (b0.04 -> 2000 poisons).  Costs ~1.5-2x crafting time.
# gradmatch only -- ignored when ATTACK=fc.
LOWMEM=""
CRAFT_BATCH=256

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# 0.04 0.02 0.01 0.005 0.002 0.001
# for bug in 0.04 0.02 0.01; do
for bug in 0.005; do
# for bug in 0.005 0.002 0.001; do

python final.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --model "$MODEL" --attack "$ATTACK" --base "$BASE" \
    --class_pair "$CLASS_PAIR" --pair_order "$PAIR_ORDER" \
    --budget "$bug" --epsilon "$EPSILON" \
    --craft_steps "$CRAFT_STEPS" --craft_alpha "$CRAFT_ALPHA" \
    --restarts "$RESTARTS" --craft_ensemble "$CRAFT_ENSEMBLE" $FAST \
    $LOWMEM --craft_batch "$CRAFT_BATCH" \
    --base_dist "$BASE_DIST" --lambda_margin "$LAMBDA" \
    --num_surrogates "$NUM_SURROGATES" --surrogate_epochs "$SURROGATE_EPOCHS" \
    --surrogate_decay $SURROGATE_DECAY \
    --num_targets "$NUM_TARGETS" --target_select "$TARGET_SELECT" \
    --num_victims "$NUM_VICTIMS" --victim_epochs "$VICTIM_EPOCHS" \
    --victim_lr "$VICTIM_LR" --victim_bs "$VICTIM_BS" --victim_decay $VICTIM_DECAY \
    --victim_wd 0.0 --clean_baseline \
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR" \
    --no_resume --recompute_deltas

done



# 40%(6/15) asr, FC, dog-bird, easyineess 10, random select, budget 0.005, resnet
# 68.8% asr, FC, dog-bird, easyineess 10, random select, budget 0.05, resnet

# 60%(9/15) asr, FC, dog-bird, easyineess 10, ours select, budget 0.005, resnet
# 50% (8/16) asr, FC, dog-bird, easyineess 10, ours select, budget 0.05, resnet

