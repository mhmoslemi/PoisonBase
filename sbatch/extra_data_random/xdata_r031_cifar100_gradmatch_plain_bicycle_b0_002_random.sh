#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_r031_cifar100_gradmatch_plain_bicycle_b0_002_random
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_r031_cifar100_gradmatch_plain_bicycle_b0_002_random-%j.out

# One extra-data.tex result cell: CIFAR100 / ResNet18BN / plain-bicycle /
# budget 0.002 / gradmatch / Random, with Jacobian disabled.
# L40S estimate 3:00 for the expanded protocol; request includes the 0:45 cushion plus a 0:15 Vulcan buffer.

export DATASET=CIFAR100
export MODEL=ResNet18BN
export CLASS_PAIR=plain-bicycle
export BUDGET=0.002
export ATTACK=gradmatch
export TARGET_FILE=target_sets/xdata_CIFAR100_ResNet18BN_plain-bicycle.json
export SURROGATE_CACHE=CIFAR100_ResNet18BN_60ep_lr0.1_bs128_seed42
export VICTIM_CACHE=CIFAR100_ResNet18BN_50ep_lr0.1_bs125_wd0_seed42
export VICTIM_LR=0.1
export CRAFT_LOWMEM=0
export CRAFT_BATCH=256
export SELECTION=random

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_job_common.sh
