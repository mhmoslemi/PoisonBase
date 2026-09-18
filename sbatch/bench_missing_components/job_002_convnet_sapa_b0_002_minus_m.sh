#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=bmc02_conv_sa_b0002_m
#SBATCH --time=0-03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/bmc02_conv_sa_b0002_m-%j.out

export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export MODEL=ConvNetBN
export ATTACK=sapa
export BUDGETS=0.002
export SELECT=minus-m
source "$SOURCE_ROOT/sbatch/bench_missing_components/_job_common.sh"
