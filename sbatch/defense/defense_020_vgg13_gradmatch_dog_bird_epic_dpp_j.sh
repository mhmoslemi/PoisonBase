#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=defense_020_vgg13_gradmatch_dog_bird_epic_dpp_j
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/defense_020_vgg13_gradmatch_dog_bird_epic_dpp_j-%j.out
#SBATCH --mail-user=mhmoslemi2338@gmail.com
#SBATCH --mail-type=ALL

# Estimated L40S runtime: 0-02:15:00; requested walltime adds 00:45.
# Source command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="gradmatch" BUDGETS="0.005" SELS="dpp" SEL_ALPHA=2.0 DEFENSES="epic" NUM_TARGETS=5 NUM_VICTIMS=4 sh defense.sh
# Effective command: USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="gradmatch" BUDGETS="0.005" SELS="dpp" SEL_ALPHA=2.0 DEFENSES="epic" NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh

export JOB_KIND=defense
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=1 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR="dog-bird" MODEL="VGG13BN" ATTACK="gradmatch" BUDGETS="0.005" SELS="dpp" SEL_ALPHA=2.0 DEFENSES="epic" NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh'
export USE_JACOBIAN_SCORE=1
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=VGG13BN
export ATTACK=gradmatch
export BUDGETS=0.005
export SELS=dpp
export SEL_ALPHA=2.0
export DEFENSES=epic
export TARGET_SELECT=''
export NUM_TARGETS=7
export NUM_VICTIMS=5
export EPIC_SUBSET=''
export NOISE_EPS=''
export FRIENDLY_CLAMP=''

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
