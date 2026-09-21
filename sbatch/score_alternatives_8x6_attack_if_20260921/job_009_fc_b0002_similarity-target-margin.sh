#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ascore009_fc_b0002_similarity-target-margin
#SBATCH --time=03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/home/mmoslem3/scratch/attack_if
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/ascore009_fc_b0002_similarity-target-margin-%j.out

# One experiment: ConvNetBN / dog-bird / fc / rho=0.002
# New selector: similarity-target-margin; exactly 8 targets x 6 victims.
export ROOT=/home/mmoslem3/scratch/attack_if
export ENV_ACTIVATE=/home/mmoslem3/ENV/bin/activate
export DATA_ROOT=/home/mmoslem3/scratch/data
export SCORE_JOB_ID=009
source "$ROOT/sbatch/score_alternatives_8x6_attack_if_20260921/_job_common.sh"
