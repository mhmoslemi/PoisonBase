#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=defense_004_convnet_fc_dog_bird_b0_02_epic_random_std
#SBATCH --time=0-02:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/defense_004_convnet_fc_dog_bird_b0_02_epic_random_std-%j.out
#SBATCH --mail-user=mhmoslemi2338@gmail.com
#SBATCH --mail-type=ALL

# L40S walltime: estimated 0-01:30:00; includes the 00:45 cushion.
# Grouped source command: USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="fc" BUDGETS="0.01 0.02" SELS="random ours dpp" SEL_ALPHA=2.0 DEFENSES="epic" NUM_TARGETS=5 NUM_VICTIMS=4 sh defense.sh
# This-cell command: USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="fc" BUDGETS=0.02 SELS=random SEL_ALPHA=2.0 DEFENSES="epic" NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh

export JOB_KIND=defense
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR="dog-bird" MODEL="ConvNetBN" ATTACK="fc" BUDGETS=0.02 SELS=random SEL_ALPHA=2.0 DEFENSES="epic" NUM_TARGETS=7 NUM_VICTIMS=5 sh defense.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ConvNetBN
export ATTACK=fc
export BUDGETS=0.02
export SELS=random
export SEL_ALPHA=2.0
export DEFENSES=epic
export TARGET_SELECT=''
export NUM_TARGETS=7
export NUM_VICTIMS=5
export EPIC_SUBSET=''
export NOISE_EPS=''
export FRIENDLY_CLAMP=''

source /home/mmoslem3/scratch/attack_if/sbatch/_job_common.sh
