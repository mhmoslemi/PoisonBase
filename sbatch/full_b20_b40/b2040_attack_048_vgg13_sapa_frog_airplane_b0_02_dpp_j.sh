#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=b2040_attack_048_vgg13_sapa_frog_airplane_b0_02_dpp_j
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/b2040_attack_048_vgg13_sapa_frog_airplane_b0_02_dpp_j-%j.out

# L40S walltime: estimated 0-07:15:00; long attack gets its full estimate plus the 00:45 cushion.
# Exactly one table cell: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=frog-airplane MODEL=VGG13BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.02 SELECT=dpp SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh

export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=frog-airplane MODEL=VGG13BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.02 SELECT=dpp SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=1
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=frog-airplane
export MODEL=VGG13BN
export ATTACK=sapa
export BUDGETS=0.02
export SELECT=dpp
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export NUM_TARGETS=8
export NUM_VICTIMS=6

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
