#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/home/mmoslem3/scratch/attack_if/ours_result/logs/slurm/pilot_CIFAR10_ConvNetBN_GRAD_random_frog-airplane_budget0.05_seed42.log
#SBATCH --job-name=pil_Conv_GR
set -e
source /home/mmoslem3/ENV/bin/activate
cd /home/mmoslem3/scratch/attack_if
python main_new.py --model ConvNetBN --attack GRAD --base random \
    --class_pair frog-airplane --budget 0.05 \
    --dataset CIFAR10 --seed 42 --num_victims 3 --num_targets 5 \
    --victim_epochs 60 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 45 --no_augmentation \
    --num_surrogate 5 --coef 1.0 --epsilon 0.0313 --craft_steps 250 --craft_lr 0.01 \
    --ref_model ResNet20BN --target_margin_low 0.05 --target_margin_high 0.5 \
    --cache_dir /home/mmoslem3/scratch/attack_if/cache --data_path /home/mmoslem3/scratch/data --out_dir /home/mmoslem3/scratch/attack_if/ours_result
