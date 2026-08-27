#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=augpre_10_r07_a025
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/%x_%j.out

# Extend an existing alpha poison cache so the five-way target intersection
# reproduces the five targets used by the completed augmentation cells.
export EXTRA_ALPHA=0.25
export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR=dog-bird MODEL=ResNet20BN ATTACK=gradmatch TARGET_SELECT=14 BUDGETS=0.002 SELECT=dpp SEL_ALPHA=0.25 NUM_TARGETS=6 NUM_VICTIMS=1 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ResNet20BN
export ATTACK=gradmatch
export BUDGETS=0.002
export SELECT=dpp
export SEL_ALPHA=0.25
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=14
export NUM_TARGETS=6
export NUM_VICTIMS=1
export PREREQ_ROW=07
export EXPECTED_RUN_NAME='CIFAR10_ResNet20BN_gradmatch_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.25_ce5_tgt14'
export REQUIRED_CACHED_TARGETS=6

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source "$SOURCE_ROOT/sbatch/_defense_extra_job_common.sh"

