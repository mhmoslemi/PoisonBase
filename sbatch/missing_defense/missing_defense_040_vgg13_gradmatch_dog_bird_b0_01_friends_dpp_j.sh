#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=missing_defense_040_vgg13_gradmatch_dog_bird_b0_01_friends_dpp_j
#SBATCH --time=0-02:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/missing_defense_040_vgg13_gradmatch_dog_bird_b0_01_friends_dpp_j-%j.out

# L40S walltime: estimated 0-02:00:00; includes the 00:45 cushion.
# Grouped source command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=gradmatch BUDGETS=0.01 SELS=dpp SEL_ALPHA=2.0 DEFENSES=friends NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh
# This-cell command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=gradmatch BUDGETS=0.01 SELS=dpp SEL_ALPHA=2.0 DEFENSES=friends NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh

export JOB_KIND=defense
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=gradmatch BUDGETS=0.01 SELS=dpp SEL_ALPHA=2.0 DEFENSES=friends NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh'
export USE_JACOBIAN_SCORE=1
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=VGG13BN
export ATTACK=gradmatch
export BUDGETS=0.01
export SELS=dpp
export SEL_ALPHA=2.0
export DEFENSES=friends
export TARGET_SELECT=''
export NUM_TARGETS=7
export NUM_VICTIMS=5
export EPIC_SUBSET=''
export NOISE_EPS=''
export FRIENDLY_CLAMP=''

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
