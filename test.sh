#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_r004_svhn_fc_8_3_b0_002_random
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_r004_svhn_fc_8_3_b0_002_random-%j.out

# One extra-data.tex result cell: SVHN / ConvNetBN / 8-3 /
# budget 0.002 / fc / Random, with Jacobian disabled.
# L40S estimate 1:30 for the expanded protocol; request includes the 0:45 cushion plus a 0:15 Vulcan buffer.

export DATASET=SVHN
export MODEL=ConvNetBN
export CLASS_PAIR=8-3
export BUDGET=0.002
export ATTACK=fc
export TARGET_FILE=target_sets/xdata_SVHN_ConvNetBN_8-3.json
export SURROGATE_CACHE=SVHN_ConvNetBN_60ep_lr0.1_bs128_seed42
export VICTIM_CACHE=SVHN_ConvNetBN_50ep_lr0.01_bs125_wd0_seed42
export VICTIM_LR=0.01
export CRAFT_LOWMEM=0
export CRAFT_BATCH=256
export SELECTION=random

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_job_common.sh
