#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_pre_tiny
#SBATCH --time=0-02:30:00
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_pre_tiny_%a-%A.out

export DATASET=TinyImageNet
export MODEL=ResNet18BN
export SURROGATE_CACHE=TinyImageNet_ResNet18BN_60ep_lr0.1_bs128_seed42
export VICTIM_CACHE=TinyImageNet_ResNet18BN_50ep_lr0.1_bs125_wd0_seed42
export VICTIM_LR=0.1

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_precompute_common.sh
