#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=score039_sapa_b0005_similarity-target-margin
#SBATCH --time=03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/score039_sapa_b0005_similarity-target-margin-%j.out

# One experiment: ConvNetBN / dog-bird / sapa / rho=0.005
# New selector: similarity-target-margin; exactly 8 targets x 6 victims.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export SCORE_JOB_ID=039
source "$ROOT/sbatch/score_alternatives_8x6_20260921/_job_common.sh"
