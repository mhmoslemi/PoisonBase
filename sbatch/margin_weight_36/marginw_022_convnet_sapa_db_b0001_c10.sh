#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=marginw_022_convnet_sapa_db_b0001_c10
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/marginw_022_convnet_sapa_db_b0001_c10-%j.out

# ours, Jacobian off; score = z(cosine distance) + 10*z(margin).
# Protocol: ConvNetBN, sapa, dog-bird, budget 0.001, 6 targets x 6 victims.
export ATTACK=sapa
export BUDGETS=0.001
export LAMBDA_MARGIN=10

source /home/mmoslem3/scratch/attack_if/sbatch/margin_weight_36/_job.sh

