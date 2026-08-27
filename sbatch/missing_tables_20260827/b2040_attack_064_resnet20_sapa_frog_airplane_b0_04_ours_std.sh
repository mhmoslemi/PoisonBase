#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=b2040_attack_064_resnet20_sapa_frog_airplane_b0_04_ours_std
#SBATCH --time=0-09:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/b2040_attack_064_resnet20_sapa_frog_airplane_b0_04_ours_std-%j.out

# L40S walltime: estimated 0-08:30:00; long attack gets its full estimate plus the 00:45 cushion.
# Exactly one table cell: USE_JACOBIAN_SCORE=0 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=frog-airplane MODEL=ResNet20BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.04 SELECT=ours SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh

export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 JACOBIAN_WEIGHT=1.0 JACOBIAN_BATCH_SIZE=64 CLASS_PAIR=frog-airplane MODEL=ResNet20BN ATTACK=sapa SHARP_MODE=worst SHARP_SIGMA=0.05 BUDGETS=0.04 SELECT=ours SEL_ALPHA=2.0 NUM_TARGETS=8 NUM_VICTIMS=6 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=frog-airplane
export MODEL=ResNet20BN
export ATTACK=sapa
export BUDGETS=0.04
export SELECT=ours
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export NUM_TARGETS=8
export NUM_VICTIMS=6

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export LEGACY_SOURCE_ROOT=/home/mmoslem3/scratch/attack_if
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source /home/mmoslem3/scratch/PoisonBase/sbatch/_job_common.sh
