#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=defextra_a01_055_vgg13_gm_b001_nodef
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/defextra_a01_055_vgg13_gm_b001_nodef-%j.out

# Exactly one defense-extra table cell: alpha=0.1, No Defense,
# VGG13BN / gradmatch / dog-bird / budget 0.01.
# Protocol: 5 targets x 1 victim; walltime includes a 00:45 cushion.

export EXTRA_ALPHA=0.1
export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=gradmatch TARGET_SELECT=50 BUDGETS=0.01 SELECT=dpp SEL_ALPHA=0.1 NUM_TARGETS=5 NUM_VICTIMS=1 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=VGG13BN
export ATTACK=gradmatch
export BUDGETS=0.01
export SELECT=dpp
export SEL_ALPHA=0.1
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=50
export NUM_TARGETS=5
export NUM_VICTIMS=1

source /home/mmoslem3/scratch/attack_if/sbatch/_defense_extra_job_common.sh
