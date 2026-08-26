#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_051_tiny_gradmatch_n01443537_n01629819_b0_001_greedy
#SBATCH --time=0-04:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/xdata_051_tiny_gradmatch_n01443537_n01629819_b0_001_greedy-%j.out

# One extra-data.tex result cell: TinyImageNet / ResNet18BN / n01443537-n01629819 /
# budget 0.001 / gradmatch / Greedy, with Jacobian disabled.
# L40S estimate 3:45; request adds the required 0:45 cushion.

export DATASET=TinyImageNet
export MODEL=ResNet18BN
export CLASS_PAIR=n01443537-n01629819
export BUDGET=0.001
export ATTACK=gradmatch
export TARGET_FILE=target_sets/appx_tiny_ResNet18BN_n01443537-n01629819.json
export SURROGATE_CACHE=TinyImageNet_ResNet18BN_60ep_lr0.1_bs128_seed42
export VICTIM_CACHE=TinyImageNet_ResNet18BN_50ep_lr0.1_bs125_wd0_seed42
export VICTIM_LR=0.1
export CRAFT_LOWMEM=1
export CRAFT_BATCH=256

source /home/mmoslem3/scratch/attack_if/sbatch/_extra_data_job_common.sh
