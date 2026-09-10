#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=marginwpb_031_convnet_sapa_db_b0002_c0p1
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/marginwpb_031_convnet_sapa_db_b0002_c0p1-%j.out

# ours, Jacobian off; score = z(cosine distance) + 0.1*z(margin).
# Protocol: ConvNetBN, sapa, dog-bird, budget 0.002, 6 targets x 6 victims.
export ATTACK=sapa
export BUDGETS=0.002
export LAMBDA_MARGIN=0.1

source /home/mmoslem3/scratch/PoisonBase/sbatch/margin_weight_36_poisonbase/_job.sh

