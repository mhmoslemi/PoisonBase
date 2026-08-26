#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=attack_030_vgg13_fc_dog_bird_ours_j
#SBATCH --time=0-07:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/attack_030_vgg13_fc_dog_bird_ours_j-%j.out
#SBATCH --mail-user=mhmoslemi2338@gmail.com
#SBATCH --mail-type=ALL

# Estimated L40S runtime: 0-06:30:00; requested walltime adds 00:45.
# Source command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="fc" BUDGETS="0.001 0.002 0.005" SELECT="ours" sh sel_dpp.sh
# Effective command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="fc" BUDGETS="0.001 0.002 0.005" SELECT="ours" sh sel_dpp.sh

export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="fc" BUDGETS="0.001 0.002 0.005" SELECT="ours" sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=1
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=VGG13BN
export ATTACK=fc
export BUDGETS='0.001 0.002 0.005'
export SELECT=ours
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
