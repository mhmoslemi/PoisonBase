#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=dmcoefpb_002_convnet_gm_db_b0001_c025
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/dmcoefpb_002_convnet_gm_db_b0001_c025-%j.out

# ours, Jacobian off; score = 0.25*z(cosine distance) + (1-0.25)*z(margin).
# Protocol: ConvNetBN, gradmatch, dog-bird, budget 0.001, 6 targets x 6 victims.
export ATTACK=gradmatch
export BUDGETS=0.001
export DISTANCE_MARGIN_COEF=0.25

source /home/mmoslem3/scratch/PoisonBase/sbatch/distance_margin_coef_30_poisonbase/_job.sh

