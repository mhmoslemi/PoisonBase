#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=dmcoefpb_019_convnet_sapa_db_b0001_c075
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/dmcoefpb_019_convnet_sapa_db_b0001_c075-%j.out

# ours, Jacobian off; score = 0.75*z(cosine distance) + (1-0.75)*z(margin).
# Protocol: ConvNetBN, sapa, dog-bird, budget 0.001, 6 targets x 6 victims.
export ATTACK=sapa
export BUDGETS=0.001
export DISTANCE_MARGIN_COEF=0.75

source /home/mmoslem3/scratch/PoisonBase/sbatch/distance_margin_coef_30_poisonbase/_job.sh

