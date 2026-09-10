#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=marginw_005_convnet_gm_db_b0001_c200
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/marginw_005_convnet_gm_db_b0001_c200-%j.out

# ours, Jacobian off; score = z(cosine distance) + 200*z(margin).
# Protocol: ConvNetBN, gradmatch, dog-bird, budget 0.001, 6 targets x 6 victims.
export ATTACK=gradmatch
export BUDGETS=0.001
export LAMBDA_MARGIN=200

source /home/mmoslem3/scratch/attack_if/sbatch/margin_weight_36/_job.sh

