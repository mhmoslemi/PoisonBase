#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xdata_043_svhn_sapa_9_2_b0_01_greedy
#SBATCH --time=0-07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xdata_043_svhn_sapa_9_2_b0_01_greedy-%j.out

# One extra-data.tex result cell: SVHN / ConvNetBN / 9-2 /
# budget 0.01 / sapa / Greedy, with Jacobian disabled.
# L40S estimate 6:00 for the expanded protocol; request includes the 0:45 cushion plus a 0:15 Vulcan buffer.

export DATASET=SVHN
export MODEL=ConvNetBN
export CLASS_PAIR=9-2
export BUDGET=0.01
export ATTACK=sapa
export TARGET_FILE=target_sets/xdata_SVHN_ConvNetBN_9-2.json
export SURROGATE_CACHE=SVHN_ConvNetBN_60ep_lr0.1_bs128_seed42
export VICTIM_CACHE=SVHN_ConvNetBN_50ep_lr0.01_bs125_wd0_seed42
export VICTIM_LR=0.01
export CRAFT_LOWMEM=1
export CRAFT_BATCH=256

source /home/mmoslem3/scratch/PoisonBase/sbatch/_extra_data_job_common.sh
