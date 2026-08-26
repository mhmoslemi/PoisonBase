#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=attack_074_vgg13_gradmatch_dog_bird_b0_005_ours_std
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/attack_074_vgg13_gradmatch_dog_bird_b0_005_ours_std-%j.out

# L40S walltime: estimated 0-03:15:00; includes the 00:45 cushion.
# Grouped source command: USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="gradmatch" BUDGETS="0.001 0.002 0.005 0.01" SELECT="ours" sh sel_dpp.sh
# This-cell command: USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="gradmatch" BUDGETS=0.005 SELECT="ours" sh sel_dpp.sh

export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="gradmatch" BUDGETS=0.005 SELECT="ours" sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=VGG13BN
export ATTACK=gradmatch
export BUDGETS=0.005
export SELECT=ours
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
