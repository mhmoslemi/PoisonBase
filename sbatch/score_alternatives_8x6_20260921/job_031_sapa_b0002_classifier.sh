#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=score031_sapa_b0002_classifier
#SBATCH --time=03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/PoisonBase
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/score031_sapa_b0002_classifier-%j.out

# One experiment: ConvNetBN / dog-bird / sapa / rho=0.002
# New selector: classifier; exactly 8 targets x 6 victims.
export ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export SCORE_JOB_ID=031
source "$ROOT/sbatch/score_alternatives_8x6_20260921/_job_common.sh"
