#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_pin_tiny
#SBATCH --array=0
#SBATCH --time=0-00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_pin_tiny_%a-%j.out

export DATASET=TinyImageNet
export MODEL=ResNet18BN
export NUM_TARGETS=4
export NUM_VICTIMS=4
export VICTIM_CACHE=TinyImageNet_ResNet18BN_50ep_lr0.1_bs125_wd0_seed42
export VICTIM_LR=0.1
PAIRS=(n01443537-n01629819)
TARGET_FILES=(target_sets/appx_tiny_ResNet18BN_n01443537-n01629819.json)

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_pin_targets_common.sh
