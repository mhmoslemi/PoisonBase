#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=bmc12_r20_sa_b0005_gigt
#SBATCH --time=0-03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/bmc12_r20_sa_b0005_gigt-%j.out

export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export MODEL=ResNet20BN
export ATTACK=sapa
export BUDGETS=0.005
export SELECT=exact
source "$SOURCE_ROOT/sbatch/bench_missing_components/_job_common.sh"
