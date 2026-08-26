#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=attack_015_convnet_sapa_dog_bird_b0_01_ours_j
#SBATCH --time=0-05:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/attack_015_convnet_sapa_dog_bird_b0_01_ours_j-%j.out
#SBATCH --mail-user=mhmoslemi2338@gmail.com
#SBATCH --mail-type=ALL

# L40S walltime: estimated 0-04:30:00; includes the 00:45 cushion.
# Grouped source command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="sapa" SHARP_MODE="worst" SHARP_SIGMA="0.05" BUDGETS="0.001 0.005 0.01 0.02 0.04" SELECT="ours" sh sel_dpp.sh
# This-cell command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="sapa" SHARP_MODE="worst" SHARP_SIGMA="0.05" BUDGETS=0.01 SELECT="ours" sh sel_dpp.sh

export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="sapa" SHARP_MODE="worst" SHARP_SIGMA="0.05" BUDGETS=0.01 SELECT="ours" sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=1
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ConvNetBN
export ATTACK=sapa
export BUDGETS=0.01
export SELECT=ours
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
