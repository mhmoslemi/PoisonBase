#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=b2040_attack_069_vgg13_sapa_dog_bird_b0_04_dpp_j
#SBATCH --time=1-05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/b2040_attack_069_vgg13_sapa_dog_bird_b0_04_dpp_j-%j.out

# L40S walltime: estimated 1-04:45:00; long attack gets its full estimate plus the 00:45 cushion.
# Grouped source command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.04 SELECT=dpp SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh
# This-cell command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.04 SELECT=dpp SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh

export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.04 SELECT=dpp SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=1
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=VGG13BN
export ATTACK=sapa
export BUDGETS=0.04
export SELECT=dpp
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export NUM_TARGETS=8
export NUM_VICTIMS=6

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
