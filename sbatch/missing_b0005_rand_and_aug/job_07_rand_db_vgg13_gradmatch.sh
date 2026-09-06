#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=b0005r_db_vgg_gm
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/%x-%j.out

export MODEL=VGG13BN ATTACK=gradmatch CLASS_PAIR=dog-bird
source /home/mmoslem3/scratch/attack_if/sbatch/missing_b0005_rand_and_aug/_random_common.sh
