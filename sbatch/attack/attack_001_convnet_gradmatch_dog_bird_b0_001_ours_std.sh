#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=attack_001_convnet_gradmatch_dog_bird_b0_001_ours_std
#SBATCH --time=0-01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/attack_001_convnet_gradmatch_dog_bird_b0_001_ours_std-%j.out

# L40S walltime: estimated 0-01:00:00; includes the 00:45 cushion.
# Grouped source command: USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="gradmatch" BUDGETS="0.001 0.005 0.01 0.02 0.04" SELECT="ours" sh sel_dpp.sh
# This-cell command: USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="gradmatch" BUDGETS=0.001 SELECT="ours" sh sel_dpp.sh

export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="gradmatch" BUDGETS=0.001 SELECT="ours" sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ConvNetBN
export ATTACK=gradmatch
export BUDGETS=0.001
export SELECT=ours
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
