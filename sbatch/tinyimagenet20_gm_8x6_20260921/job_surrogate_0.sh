#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=tiny20_surrogate_0
#SBATCH --time=03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/tiny20_surrogate_0-%j.out

export PHASE=surrogate
export PHASE_ID=0
source /home/mmoslem3/scratch/PoisonBase/sbatch/tinyimagenet20_gm_8x6_20260921/_job_common.sh
