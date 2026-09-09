#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=dmcoef_014_convnet_gm_db_b0002_c075
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/dmcoef_014_convnet_gm_db_b0002_c075-%j.out

# ours, Jacobian off; score = 0.75*z(cosine distance) + (1-0.75)*z(margin).
# Protocol: ConvNetBN, gradmatch, dog-bird, budget 0.002, 6 targets x 6 victims.
export ATTACK=gradmatch
export BUDGETS=0.002
export DISTANCE_MARGIN_COEF=0.75

source /home/mmoslem3/scratch/attack_if/sbatch/distance_margin_coef_30/_job.sh

