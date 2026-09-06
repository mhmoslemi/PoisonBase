#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=b0005r_fa_vgg_fc
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/%x-%j.out

export MODEL=VGG13BN ATTACK=fc CLASS_PAIR=frog-airplane
source /home/mmoslem3/scratch/attack_if/sbatch/missing_b0005_rand_and_aug/_random_common.sh
