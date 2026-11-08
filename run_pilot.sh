#!/bin/bash
MODEL=$1
ATTACK=$2
BASE=$3
CLASS_PAIR=$4
LO=$5
HI=$6
source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if
python main_new.py --model "$MODEL" --attack "$ATTACK" --base "$BASE" \
    --class_pair "$CLASS_PAIR" --budget 0.05 \
    --dataset CIFAR10 --seed 42 --num_victims 3 --num_targets 5 \
    --victim_epochs 60 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 45 --no_augmentation \
    --num_surrogate 5 --coef 1.0 --epsilon 0.0313 --craft_steps 250 --craft_lr 0.01 \
    --ref_model ResNet20BN --target_margin_low "$LO" --target_margin_high "$HI" \
    --cache_dir ./cache --data_path /home/mmoslem3/scratch/data --out_dir ours_result
echo "===JOB_DONE==="
