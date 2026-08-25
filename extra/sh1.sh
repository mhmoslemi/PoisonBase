#!/usr/bin/env bash
#
# RESUME  CIFAR10_VGG13BN_gradmatch_random_dog-bird_b0.02_eps8_seed42_ce5_tgt50
#         (was sh3.sh -- killed by the Slurm TIME LIMIT on kn172 at 20:11)
#
# Targets (10): [630, 409, 2270, 9870, 1656, 5022, 9090, 6033, 7940, 9731]
#   done   -> 630 409 2270 9870 1656 5022 9090 6033
#             (8/10 complete, 48 trials in results.csv)
#   to run -> 7940 9731
#             (2 targets x 6 victims = 12 trainings)
#   9731 was interrupted during crafting, so its delta is NOT in deltas.pt
#   and it gets crafted again from scratch.
#
# Rough cost: 2 crafts x ~23-39 min + 12 victims x ~1.4 min
#             = ~1h02m to ~1h34m   (craft time varies a lot with the node)
#
# How the resume works (nothing in final.py needed changing):
#   * --no_resume is NOT passed, so results.csv is read first and every
#     (target, victim) pair already in it is skipped, then new rows are appended.
#   * --recompute_deltas is NOT passed, so deltas.pt is reloaded and the 8
#     finished targets skip crafting entirely instead of redoing it.
# Everything else is byte-identical to the original command.

DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
OUT_DIR=ours_result
CACHE_DIR=./cache
SEED=42

MODEL=VGG13BN
ATTACK=gradmatch
BASE=random
CLASS_PAIR=dog-bird
PAIR_ORDER=poison-target

BUDGET=0.02
EPSILON=0.0313725         # 8/255
CRAFT_STEPS=250
CRAFT_ALPHA=0.0039216     # 1/255, signed-Adam lr
RESTARTS=8
CRAFT_ENSEMBLE=5

TARGET_SELECT=50
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

# NOTE  resuming is idempotent -- if this dies again (time limit, Ctrl-C), just
#       rerun the same script. Every finished (target, victim) is already durable
#       in results.csv and every finished craft is in deltas.pt.
#
# NOTE  the order the 10 targets are listed in can differ between invocations
#       (their p_adv scores are tied at ~0.0000, so the ranking is not stable).
#       Harmless: resume is keyed on target_idx, not position, and the SET of 10
#       is always the same.
#
# NOTE  the final "target-prediction tally" line counts only the victims trained
#       in THIS invocation (12), not all 60 -- results.csv stores success/fail but
#       not the predicted class, so the tally cannot be rebuilt for the targets
#       finished earlier. ASR and CTA in the closing "====" line ARE computed over
#       the full results.csv, so those are correct for all 10.
