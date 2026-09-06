#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=seldpp9_convnet_sapa
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/seldpp9_convnet_sapa-%j.out

export MODEL=ConvNetBN
export ATTACK=sapa
export CLASS_PAIR=dog-bird
source /home/mmoslem3/scratch/attack_if/sbatch/sel_dpp_9/_job_common.sh
