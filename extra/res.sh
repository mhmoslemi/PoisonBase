#!/usr/bin/env bash
#
# Resume the 5 runs in result_ours.txt that still have no "====" summary.
#
#   VGG13BN    / frog-airplane / b0.04  / tgt3    9/10 targets, 54 trials -- LAST TARGET ONLY
#   VGG13BN    / dog-bird      / b0.04  / tgt3    9/10 targets, 54 trials -- LAST TARGET ONLY
#   ResNet20BN / frog-airplane / b0.002 / tgt2    8/10 targets, 50 trials (2672 has 2/6)
#   ResNet20BN / frog-airplane / b0.005 / tgt2    8/10 targets, 50 trials (2672 has 2/6)
#   ResNet20BN / frog-airplane / b0.04  / tgt2    7/10 targets, 47 trials (4490 has 5/6)
#
# The 4 that this script finished earlier (ConvNetBN dog-bird b0.02, and
# ResNet20BN dog-bird b0.002 / b0.005 / b0.04) are done and have been removed.
#
# Every argument is copied verbatim from each run's own "args:" line in
# result_ours.txt. --no_resume and --recompute_deltas are the only two that are
# dropped; that is what makes these resume (skip finished (target, victim) pairs,
# reuse finished crafts) instead of starting over.

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

# identical across all 7 runs
COMMON="--dataset CIFAR10
        --data_path /home/mmoslem3/scratch/data
        --seed 42
        --cache_dir ./cache
        --out_dir ours_result
        --dsa_strategy color_crop_cutout_flip_scale_rotate
        --attack fc
        --base ours
        --pair_order poison-target
        --num_poisons 500
        --epsilon 0.0313725
        --craft_steps 250
        --craft_alpha 0.0039216
        --restarts 8
        --fc_restarts 1
        --fc_mode sample
        --craft_ensemble 5
        --craft_batch 256
        --lambda_margin 1.0
        --base_dist cosine
        --num_surrogates 5
        --surrogate_epochs 60
        --surrogate_lr 0.1
        --surrogate_bs 128
        --surrogate_decay 35 45
        --surrogate_wd 0.0
        --num_targets 10
        --victim_epochs 50
        --victim_lr 0.1
        --victim_bs 125
        --victim_decay 40
        --victim_wd 0.0
        --num_victims 6
        --clean_baseline
        --precompute_part both"

# model  class_pair  budget  target_select  target_idx_file
for rec in \
  "VGG13BN frog-airplane 0.04 3 target_sets/VGG13BN_fc_frog-airplane.json" \
  "VGG13BN dog-bird 0.04 3 target_sets/VGG13BN_fc_dog-bird.json" \
  "ResNet20BN frog-airplane 0.002 2 target_sets/ResNet20BN_fc_frog-airplane.json" \
  "ResNet20BN frog-airplane 0.005 2 target_sets/ResNet20BN_fc_frog-airplane.json" \
  "ResNet20BN frog-airplane 0.04 2 target_sets/ResNet20BN_fc_frog-airplane.json"
do
    set -- $rec
    echo "=== resuming $1 / $2 / b$3 / tgt$4 ==="
    python final.py $COMMON \
        --model "$1" --class_pair "$2" --budget "$3" \
        --target_select "$4" --target_idx_file "$5"
done
