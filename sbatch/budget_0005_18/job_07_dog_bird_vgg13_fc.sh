#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=b0_0005_dog_bird_vgg13_fc
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/b0_0005_dog_bird_vgg13_fc-%j.out

export MODEL=VGG13BN
export ATTACK=fc
export CLASS_PAIR=dog-bird
export BUDGETS=0.0005
source /home/mmoslem3/scratch/attack_if/sbatch/budget_0005_18/_job_common.sh
