#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_pin_svhn
#SBATCH --array=0-4
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_pin_svhn_%a-%j.out

export DATASET=SVHN
export MODEL=ConvNetBN
export NUM_TARGETS=6
export NUM_VICTIMS=5
export VICTIM_CACHE=SVHN_ConvNetBN_50ep_lr0.01_bs125_wd0_seed42
export VICTIM_LR=0.01
PAIRS=(0-4 6-1 7-5 8-3 9-2)
TARGET_FILES=(
    target_sets/xdata_SVHN_ConvNetBN_0-4.json
    target_sets/xdata_SVHN_ConvNetBN_6-1.json
    target_sets/xdata_SVHN_ConvNetBN_7-5.json
    target_sets/xdata_SVHN_ConvNetBN_8-3.json
    target_sets/xdata_SVHN_ConvNetBN_9-2.json
)

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_pin_targets_common.sh
