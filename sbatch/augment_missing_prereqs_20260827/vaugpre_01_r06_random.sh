#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=vaugpre_01_r06_random
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/%x_%j.out

# Generate exactly one missing poison cache.
export JOB_KIND=attack
export EXTRA_ALPHA=2
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR=dog-bird MODEL=ResNet20BN ATTACK=fc TARGET_SELECT=10 BUDGETS=0.01 SELECT=random SEL_ALPHA=2 NUM_TARGETS=5 NUM_VICTIMS=1 RECOMPUTE_DELTAS=1 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ResNet20BN
export ATTACK=fc
export BUDGETS=0.01
export SELECT=random
export SEL_ALPHA=2
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=10
export NUM_TARGETS=5
export REQUIRED_CACHED_TARGETS=5
export NUM_VICTIMS=1
export RECOMPUTE_DELTAS=1
export PREREQ_ROW=06
export EXPECTED_RUN_NAME='CIFAR10_ResNet20BN_fc_random_dog-bird_b0.01_eps8_seed42_ce5_tgt10'

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source "$SOURCE_ROOT/sbatch/_defense_extra_job_common.sh"
