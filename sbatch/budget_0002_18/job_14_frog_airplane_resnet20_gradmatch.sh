#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=b0002_frog_airplane_resnet20_gradmatch
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/b0002_frog_airplane_resnet20_gradmatch-%j.out

export MODEL=ResNet20BN
export ATTACK=gradmatch
export CLASS_PAIR=frog-airplane
export BUDGETS=0.0002
source /home/mmoslem3/scratch/PoisonBase/sbatch/budget_0002_18/_job_common.sh
