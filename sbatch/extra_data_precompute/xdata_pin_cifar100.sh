#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_pin_cifar100
#SBATCH --array=0-4
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_pin_cifar100_%a-%j.out

export DATASET=CIFAR100
export MODEL=ResNet18BN
export NUM_TARGETS=4
export NUM_VICTIMS=4
export VICTIM_CACHE=CIFAR100_ResNet18BN_50ep_lr0.1_bs125_wd0_seed42
export VICTIM_LR=0.1
PAIRS=(bottle-road plain-bicycle sea-willow_tree sunflower-cattle wardrobe-lawn_mower)
TARGET_FILES=(
    target_sets/xdata_CIFAR100_ResNet18BN_bottle-road.json
    target_sets/xdata_CIFAR100_ResNet18BN_plain-bicycle.json
    target_sets/xdata_CIFAR100_ResNet18BN_sea-willow_tree.json
    target_sets/xdata_CIFAR100_ResNet18BN_sunflower-cattle.json
    target_sets/xdata_CIFAR100_ResNet18BN_wardrobe-lawn_mower.json
)

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_pin_targets_common.sh
