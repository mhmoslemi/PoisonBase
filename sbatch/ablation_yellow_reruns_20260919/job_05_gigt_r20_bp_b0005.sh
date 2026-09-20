#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ay05_gigt_r20_bp_b0005
#SBATCH --time=0-03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/ay05_gigt_r20_bp_b0005-%j.out

export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export MODEL=ResNet20BN
export ATTACK=fc
export BUDGETS=0.005
export SELECT=exact
source "$SOURCE_ROOT/sbatch/ablation_yellow_reruns_20260919/_job_common.sh"
