#!/usr/bin/env bash
#
# RESUME  CIFAR10_VGG13BN_gradmatch_random_frog-airplane_b0.01_eps8_seed42_ce5_tgt12
#         (the run that was interrupted in Afinal.sh)
#
# Targets: [2783, 2507, 6738, 5787, 7815, 5074, 7356, 8267, 2264, 6481]
#   done   -> 2783 2507 6738 5787 7815 5074   (6/10, all 6 victims each)
#   to run -> 7356 8267 2264 6481             (4 targets x 6 victims = 24 trainings)
#
# How the resume works (nothing in final.py needed changing):
#   * --no_resume is NOT passed, so results.csv is read first and every
#     (target, victim) pair already in it is skipped, then new rows are appended.
#   * --recompute_deltas is NOT passed, so deltas.pt is reloaded and the 6
#     finished targets skip crafting (~20 min each) instead of redoing it.
# Everything else is byte-identical to the original command.

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

MODEL=VGG13BN
ATTACK=gradmatch
BASE=random
CLASS_PAIR=frog-airplane
PAIR_ORDER=poison-target

BUDGET=0.01
EPSILON=0.0313725         # 8/255
CRAFT_STEPS=250
CRAFT_ALPHA=0.0039216     # 1/255, signed-Adam lr
RESTARTS=8
CRAFT_ENSEMBLE=5

TARGET_SELECT=12
NUM_TARGETS=10

BASE_DIST=l2
LAMBDA=1.0

NUM_SURROGATES=5
SURROGATE_EPOCHS=60
SURROGATE_DECAY="35 45"

NUM_VICTIMS=6
VICTIM_EPOCHS=50
VICTIM_LR=0.1
VICTIM_BS=125
VICTIM_DECAY="40"

# kept exactly as the interrupted run had them: --craft_lowmem takes precedence
# over --fast_gradmatch inside craft_gradmatch, so this is the exact objective.
FAST="--fast_gradmatch"
LOWMEM="--craft_lowmem"
CRAFT_BATCH=256

RUN=CIFAR10_${MODEL}_${ATTACK}_${BASE}_${CLASS_PAIR}_b${BUDGET}_eps8_seed${SEED}_ce${CRAFT_ENSEMBLE}_tgt${TARGET_SELECT}

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# ---- replay everything the lost terminal already printed --------------------
LOG="$OUT_DIR/$RUN/log.txt"
if [ -f "$LOG" ]; then
    echo "################################################################################"
    echo "# previous output, replayed from $LOG"
    echo "################################################################################"
    cat "$LOG"
    echo
    echo "################################################################################"
    echo "# end of previous log -- resuming below"
    echo "################################################################################"
    echo
else
    echo "WARNING: $LOG not found -- this will start the run from scratch."
fi

python final.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
    --model "$MODEL" --attack "$ATTACK" --base "$BASE" \
    --class_pair "$CLASS_PAIR" --pair_order "$PAIR_ORDER" \
    --budget "$BUDGET" --epsilon "$EPSILON" \
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
    --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR"

# NOTE  the final "target-prediction tally" line counts only the victims trained
#       in THIS invocation (24), not all 60 -- results.csv stores success/fail but
#       not the predicted class, so the tally cannot be rebuilt for the 6 targets
#       that were finished earlier. ASR and CTA in the closing "====" line ARE
#       computed over the full results.csv, so those are correct for all 10.
#
# NOTE  the 6 finished targets take their poison bases from bases.json instead of
#       drawing them, so the base-selection RNG sits 6 draws earlier than it would
#       in an uninterrupted run. Targets 7-10 therefore get different (still
#       uniformly random, same-class) base images than a clean rerun would pick.
