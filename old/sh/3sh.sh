#!/usr/bin/env bash
#
# RESUME  CIFAR10_ResNet20BN_gradmatch_random_frog-airplane_b0.04_eps8_seed42_ce5_tgt10
#
# Every hyperparameter below was copied verbatim out of this run's own
# "args:" line in ours_result/<run>/log.txt. The ONLY two arguments that
# differ from the original invocation are --no_resume and --recompute_deltas,
# which are omitted -- that is what makes it resume instead of restart:
#   * results.csv is read first, so the 24 (target, victim) trials already
#     done are skipped and new rows appended.
#   * deltas.pt is reloaded, so the 4 finished crafts are reused.
#
# Targets (10): [9235, 338, 255, 3207, 8563, 912, 4639, 2570, 3117, 2041]
#   done   -> 9235 338 255 3207  (4/10, 6 victims each)
#   to run -> 8563 912 4639 2570 3117 2041  (36 trainings)
#   crafts already cached for: none of the remaining targets -- each is crafted fresh
#
# Resuming is idempotent: if this dies again, just rerun it.

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# refuse to run without a GPU: final.py silently falls back to CPU, where crafting
# 2000 poisons takes weeks and looks like a hang
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: no GPU visible here (torch.cuda.is_available() == False)."
    echo "       final.py would run on CPU. Get a GPU allocation, then rerun."
    exit 1
fi

RUN=CIFAR10_ResNet20BN_gradmatch_random_frog-airplane_b0.04_eps8_seed42_ce5_tgt10
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
        --model ResNet20BN
        --seed 42
        --cache_dir ./cache
        --out_dir ours_result
        --dsa_strategy color_crop_cutout_flip_scale_rotate"

# attack
A_ATTACK="--attack gradmatch
          --base random
          --class_pair frog-airplane
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
           --target_select 10"

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
