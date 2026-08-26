#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_044_cifar100_gradmatch_sea_willow_tree_b0_005_greedy
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_044_cifar100_gradmatch_sea_willow_tree_b0_005_greedy-%j.out

# One extra-data.tex result cell: CIFAR100 / ResNet18BN / sea-willow_tree /
# budget 0.005 / gradmatch / Greedy, with Jacobian disabled.
# L40S estimate 1:45; request adds the required 0:45 cushion.

export DATASET=CIFAR100
export MODEL=ResNet18BN
export CLASS_PAIR=sea-willow_tree
export BUDGET=0.005
export ATTACK=gradmatch
export TARGET_FILE=target_sets/xdata_CIFAR100_ResNet18BN_sea-willow_tree.json
export SURROGATE_CACHE=CIFAR100_ResNet18BN_60ep_lr0.1_bs128_seed42
export VICTIM_CACHE=CIFAR100_ResNet18BN_50ep_lr0.1_bs125_wd0_seed42
export VICTIM_LR=0.1
export CRAFT_LOWMEM=1
export CRAFT_BATCH=256

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_job_common.sh
