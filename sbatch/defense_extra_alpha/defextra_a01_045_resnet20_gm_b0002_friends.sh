#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=defextra_a01_045_resnet20_gm_b0002_friends
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/defextra_a01_045_resnet20_gm_b0002_friends-%j.out

# Exactly one defense-extra table cell: alpha=0.1, FRIENDS,
# ResNet20BN / gradmatch / dog-bird / budget 0.002.
# Protocol: 5 targets x 4 victims; walltime includes a 00:45 cushion.

export EXTRA_ALPHA=0.1
export JOB_KIND=defense
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR=dog-bird MODEL=ResNet20BN ATTACK=gradmatch TARGET_SELECT=14 BUDGETS=0.002 SELS=dpp SEL_ALPHA=0.1 DEFENSES=friends NUM_TARGETS=5 NUM_VICTIMS=4 sh defense.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ResNet20BN
export ATTACK=gradmatch
export BUDGETS=0.002
export SELS=dpp
export SEL_ALPHA=0.1
export DEFENSES=friends
export TARGET_SELECT=14
export NUM_TARGETS=5
export NUM_VICTIMS=4
export EPIC_SUBSET=''
export NOISE_EPS=''
export FRIENDLY_CLAMP=''

source /home/mmoslem3/scratch/attack_if/sbatch/_defense_extra_job_common.sh
