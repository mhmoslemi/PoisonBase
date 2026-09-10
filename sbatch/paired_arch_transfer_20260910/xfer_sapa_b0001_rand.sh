#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=xfer_sapa_b0001_rand
#SBATCH --time=0-02:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/xfer_sapa_b0001_rand-%j.out

# Shared RAND is created once for this victim/attack/budget and reused for all S columns.
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
export XFER_ATTACK=sapa
export XFER_BUDGET=0.001
export XFER_SELECTION=random
export XFER_SELECTOR_MODEL=shared
export XFER_RUN_NAME=CIFAR10_ConvNetBN_sapa_random_dog-bird_b0.001_eps8_seed42_K20_worst0.05_ce5_tgt70
export ORIGINAL_COMMAND="A=V=ConvNetBN S=shared K=20 attack=sapa selection=RAND budget=0.001 targets=6 victims=6"
source "$SOURCE_ROOT/sbatch/paired_arch_transfer_20260910/_job_common.sh"
