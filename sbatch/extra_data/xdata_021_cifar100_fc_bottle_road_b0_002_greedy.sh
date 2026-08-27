#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_021_cifar100_fc_bottle_road_b0_002_greedy
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_021_cifar100_fc_bottle_road_b0_002_greedy-%j.out

# One extra-data.tex result cell: CIFAR100 / ResNet18BN / bottle-road /
# budget 0.002 / fc / Greedy, with Jacobian disabled.
# L40S estimate 0:30; request includes the 0:45 cushion plus a 0:15 Vulcan buffer.

export DATASET=CIFAR100
export MODEL=ResNet18BN
export CLASS_PAIR=bottle-road
export BUDGET=0.002
export ATTACK=fc
export TARGET_FILE=target_sets/xdata_CIFAR100_ResNet18BN_bottle-road.json
export SURROGATE_CACHE=CIFAR100_ResNet18BN_60ep_lr0.1_bs128_seed42
export VICTIM_CACHE=CIFAR100_ResNet18BN_50ep_lr0.1_bs125_wd0_seed42
export VICTIM_LR=0.1
export CRAFT_LOWMEM=0
export CRAFT_BATCH=256

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_job_common.sh
