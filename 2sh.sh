#!/usr/bin/env bash
#
# RESUME  CIFAR10_VGG13BN_gradmatch_random_dog-bird_b0.04_eps8_seed42_ce5_tgt50
#
# Every hyperparameter below was copied verbatim out of this run's own
# "args:" line in ours_result/<run>/log.txt. The ONLY two arguments that
# differ from the original invocation are --no_resume and --recompute_deltas,
# which are omitted -- that is what makes it resume instead of restart:
#   * results.csv is read first, so the 54 (target, victim) trials already
#     done are skipped and new rows appended.
#   * deltas.pt is reloaded, so the 9 finished crafts are reused.
#
# Targets (10): [630, 409, 2270, 9870, 1656, 5022, 9090, 6033, 7940, 9731]
#   done   -> 630 409 2270 9870 1656 5022 9090 6033 7940  (9/10, 6 victims each)
#   to run -> 9731  (6 trainings)
#   crafts already cached for: none of the remaining targets -- each is crafted fresh
#
# Resuming is idempotent: if this dies again, just rerun it.

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

RUN=CIFAR10_VGG13BN_gradmatch_random_dog-bird_b0.04_eps8_seed42_ce5_tgt50
LOG="ours_result/$RUN/log.txt"
if [ -f "$LOG" ]; then
    echo "###### previous output, replayed from $LOG ######"
    cat "$LOG"
    echo "###### end of previous log -- resuming below ######"
    echo
else
    echo "WARNING: $LOG not found -- this would START OVER, not resume."
    exit 1
fi

# data + model
A_DATA="--dataset CIFAR10
        --data_path /home/mmoslem3/scratch/data
        --model VGG13BN
        --seed 42
        --cache_dir ./cache
        --out_dir ours_result
        --dsa_strategy color_crop_cutout_flip_scale_rotate"

# attack
A_ATTACK="--attack gradmatch
          --base random
          --class_pair dog-bird
          --pair_order poison-target
          --budget 0.04
          --num_poisons 500
          --epsilon 0.0313725
          --craft_steps 250
          --craft_alpha 0.0039216
          --restarts 8
          --fc_restarts 1
          --fc_mode sample
          --craft_ensemble 5
          --fast_gradmatch
          --craft_lowmem
          --craft_batch 256"

# base selection
A_BASE="--lambda_margin 1.0
        --base_dist l2"

# surrogates
A_SURROGATES="--num_surrogates 5
              --surrogate_epochs 60
              --surrogate_lr 0.1
              --surrogate_bs 128
              --surrogate_decay 35 45
              --surrogate_wd 0.0"

# targets
A_TARGETS="--num_targets 10
           --target_select 50"

# victims
A_VICTIMS="--num_victims 6
           --victim_epochs 50
           --victim_lr 0.1
           --victim_bs 125
           --victim_decay 40
           --victim_wd 0.0
           --clean_baseline"

# bookkeeping
A_BOOKKEEPING="--precompute_part both"

python final.py \
    $A_DATA \
    $A_ATTACK \
    $A_BASE \
    $A_SURROGATES \
    $A_TARGETS \
    $A_VICTIMS \
    $A_BOOKKEEPING
