#!/bin/bash
# Full sweep: 3 models x 2 attacks x 2 base-selection methods x 2 class pairs = 24 runs.
# Each run trains num_targets * num_victims = 60 victims from scratch (60 epochs each),
# so this is a large compute budget. main.py checkpoints surrogates / targets / deltas /
# clean baselines to --cache_dir and appends completed (target, victim) trials to each
# run's results.csv, so a killed/restarted invocation resumes instead of recomputing from
# zero. No new CLI flags were added for the logging/baseline/summary changes below, so
# this invocation is unchanged from before -- only the outputs main.py produces changed:
#   <out_dir>/<run_name>/log.txt        -- append-mode log for that run (never truncated)
#   <out_dir>/<run_name>/summary.json   -- this run's asr_mean/std, cta_post_mean/std,
#                                           cta_baseline_mean/std
#   <out_dir>/summary_all.csv           -- one row per run, appended as each combo in the
#                                           sweep below finishes; the single place to read
#                                           final numbers from once the sweep is done
#   <cache_dir>/clean_baseline/         -- pre-attack CTA checkpoints, one set per model
#                                           architecture, shared across every attack/base/
#                                           class_pair combo for that model

set -e

DATASET=CIFAR10
DATA_PATH=./data
SEED=42

MODELS=(ConvNetBN VGG13BN ResNet20BN)
ATTACKS=(FC GRAD)
BASES=(random ours)
CLASS_PAIRS=(dog-bird frog-airplane)

# 'ours' base-selection hyperparameters
NUM_SURROGATE=5
COEF=1.0

# poisoning budget / crafting (not specified in the brief -- adjust freely)
NUM_POISONS=25
EPSILON=0.0313          # 8/255
CRAFT_STEPS=250
CRAFT_LR=0.01

# target / victim protocol
NUM_TARGETS=10
NUM_VICTIMS=6
VICTIM_EPOCHS=60
VICTIM_LR=0.1
VICTIM_BS=256
VICTIM_DECAY="35 45"

# target-difficulty band ("target easiness")
REF_MODEL=ResNet20BN
TARGET_MARGIN_LOW=0.05
TARGET_MARGIN_HIGH=0.5

CACHE_DIR=./cache
OUT_DIR=./runs

for CLASS_PAIR in "${CLASS_PAIRS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for ATTACK in "${ATTACKS[@]}"; do
      for BASE in "${BASES[@]}"; do

        echo "=== ${CLASS_PAIR} | ${MODEL} | ${ATTACK} | base=${BASE} ==="

        python main.py \
          --dataset "$DATASET" --data_path "$DATA_PATH" --seed "$SEED" \
          --model "$MODEL" --attack "$ATTACK" --base "$BASE" --class_pair "$CLASS_PAIR" \
          --num_surrogate "$NUM_SURROGATE" --coef "$COEF" \
          --num_poisons "$NUM_POISONS" --epsilon "$EPSILON" \
          --craft_steps "$CRAFT_STEPS" --craft_lr "$CRAFT_LR" \
          --num_targets "$NUM_TARGETS" --num_victims "$NUM_VICTIMS" \
          --victim_epochs "$VICTIM_EPOCHS" --victim_lr "$VICTIM_LR" --victim_bs "$VICTIM_BS" \
          --victim_decay $VICTIM_DECAY --no_augmentation \
          --ref_model "$REF_MODEL" \
          --target_margin_low "$TARGET_MARGIN_LOW" --target_margin_high "$TARGET_MARGIN_HIGH" \
          --cache_dir "$CACHE_DIR" --out_dir "$OUT_DIR"

      done
    done
  done
done