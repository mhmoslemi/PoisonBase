#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=b0_0005_frog_airplane_vgg13_sapa
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/b0_0005_frog_airplane_vgg13_sapa-%j.out

export MODEL=VGG13BN
export ATTACK=sapa
export CLASS_PAIR=frog-airplane
export BUDGETS=0.0005
source /home/mmoslem3/scratch/attack_if/sbatch/budget_0005_18/_job_common.sh
