#!/usr/bin/env bash
#
# base=ours (cosine) sweep -- shard 8 of 18.  Estimated ~5.7 h on one GPU.
#
#   ResNet20BN gradmatch dog-bird      b0.02   tgt14   5/10 targets done, 29 trials left
#   VGG13BN    fc        dog-bird      b0.01   tgt3    0/10 targets done, 60 trials left
#
# Every flag matches the random-base sweep except --base ours (plus --base_dist
# cosine and --lambda_margin, which only select_base_ours() ever reads). Targets
# are pinned with --target_idx_file to the exact 10 images the random-base run
# attacked, so the comparison stays paired.
#
# --no_resume / --recompute_deltas are deliberately NOT passed: every run here is
# resumable, so if this shard dies just rerun it and it picks up where it stopped.

source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if

COMMON="--dataset CIFAR10
        --data_path /home/mmoslem3/scratch/data
        --seed 42
        --cache_dir ./cache
        --out_dir ours_result
        --dsa_strategy color_crop_cutout_flip_scale_rotate
        --base ours
        --base_dist cosine
        --lambda_margin 1.0
        --pair_order poison-target
        --num_poisons 500
        --epsilon 0.0313725
        --craft_steps 250
        --craft_alpha 0.0039216
        --restarts 8
        --fc_restarts 1
        --fc_mode sample
        --craft_ensemble 5
        --num_surrogates 5
        --surrogate_epochs 60
        --surrogate_lr 0.1
        --surrogate_bs 128
        --surrogate_decay 35 45
        --surrogate_wd 0.0
        --num_targets 10
        --num_victims 6
        --victim_epochs 50
        --victim_lr 0.1
        --victim_bs 125
        --victim_decay 40
        --victim_wd 0.0
        --clean_baseline
        --precompute_part both"

# model  attack  pair  budget  target_select  target_idx_file  mem
for rec in \
  "ResNet20BN gradmatch dog-bird 0.02 14 target_sets/ResNet20BN_gradmatch_dog-bird.json none" \
  "VGG13BN fc dog-bird 0.01 3 target_sets/VGG13BN_fc_dog-bird.json none"
do
    set -- $rec
    MEM=""
    [ "$7" = lowmem ] && MEM="--craft_lowmem --craft_batch 256 --fast_gradmatch"
    echo "=== $1 / $2 / $3 / b$4 (tgt$5) ==="
    python final.py $COMMON \
        --model "$1" --attack "$2" --class_pair "$3" --budget "$4" \
        --target_select "$5" --target_idx_file "$6" $MEM
done
