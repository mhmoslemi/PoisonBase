#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=missing_defense_043_vgg13_sapa_dog_bird_b0_005_friends_ours_std
#SBATCH --time=0-02:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/missing_defense_043_vgg13_sapa_dog_bird_b0_005_friends_ours_std-%j.out

# L40S walltime: estimated 0-02:00:00; includes the 00:45 cushion.
# Grouped source command: USE_JACOBIAN_SCORE=0 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=sapa TARGET_SELECT=50 BUDGETS=0.005 SELS=ours SEL_ALPHA=2.0 DEFENSES=friends NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh
# This-cell command: USE_JACOBIAN_SCORE=0 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=sapa TARGET_SELECT=50 BUDGETS=0.005 SELS=ours SEL_ALPHA=2.0 DEFENSES=friends NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh

export JOB_KIND=defense
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=dog-bird MODEL=VGG13BN ATTACK=sapa TARGET_SELECT=50 BUDGETS=0.005 SELS=ours SEL_ALPHA=2.0 DEFENSES=friends NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=VGG13BN
export ATTACK=sapa
export BUDGETS=0.005
export SELS=ours
export SEL_ALPHA=2.0
export DEFENSES=friends
export TARGET_SELECT=50
export NUM_TARGETS=7
export NUM_VICTIMS=5
export EPIC_SUBSET=''
export NOISE_EPS=''
export FRIENDLY_CLAMP=''

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
