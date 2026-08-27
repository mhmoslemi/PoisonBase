#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=augpre_01_r01_a025
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/%x_%j.out

# One resumable poison-optimization prerequisite for augment-extra row 01.
export EXTRA_ALPHA=0.25
export JOB_KIND=attack
export ORIGINAL_COMMAND='USE_JACOBIAN_SCORE=0 CLASS_PAIR=dog-bird MODEL=ConvNetBN ATTACK=fc TARGET_SELECT=50 BUDGETS=0.002 SELECT=dpp SEL_ALPHA=0.25 NUM_TARGETS=7 NUM_VICTIMS=1 sh sel_dpp.sh'
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export CLASS_PAIR=dog-bird
export MODEL=ConvNetBN
export ATTACK=fc
export BUDGETS=0.002
export SELECT=dpp
export SEL_ALPHA=0.25
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=50
export NUM_TARGETS=7
export REQUIRED_CACHED_TARGETS=7
export NUM_VICTIMS=1
export PREREQ_ROW=01
export EXPECTED_RUN_NAME='CIFAR10_ConvNetBN_fc_ours_dog-bird_b0.002_eps8_seed42_lam1_cosine_seldpp0.25_ce5_tgt50'

export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/PoisonBase/data
export PYTHON_ENV=/home/mmoslem3/ENV

source "$SOURCE_ROOT/sbatch/_defense_extra_job_common.sh"
