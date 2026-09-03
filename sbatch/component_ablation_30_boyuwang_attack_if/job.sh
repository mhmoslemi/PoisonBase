#!/bin/bash
# Replacement template for the 30 -M jobs assigned to aip-boyuwang. The root
# submitter supplies one of the original 30 configurations on each submission.
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=minus_m_30_attack_if
#SBATCH --time=0-03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/attack_if/sbatch/logs/minus_m_30_attack_if-%j.out

set -Eeuo pipefail

: "${MODEL:?MODEL was not exported by the submitter}"
: "${ATTACK:?ATTACK was not exported by the submitter}"
: "${CLASS_PAIR:?CLASS_PAIR was not exported by the submitter}"
: "${BUDGETS:?BUDGETS was not exported by the submitter}"

# Exact paths for the aip-boyuwang environment. _job_common.sh activates
# $PYTHON_ENV/bin/activate before staging and launching sel_dpp.sh.
export SOURCE_ROOT=/home/mmoslem3/scratch/attack_if
export PERSIST_DATA_ROOT=/home/mmoslem3/scratch/data
export PYTHON_ENV=/home/mmoslem3/ENV

export SELECT=minus-m
export JOB_KIND=attack
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export SEL_ALPHA=2.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export NUM_TARGETS=8
export NUM_VICTIMS=5
export ORIGINAL_COMMAND="source $PYTHON_ENV/bin/activate; SOURCE_ROOT=$SOURCE_ROOT PERSIST_DATA_ROOT=$PERSIST_DATA_ROOT MODEL=$MODEL ATTACK=$ATTACK CLASS_PAIR=$CLASS_PAIR BUDGETS=$BUDGETS SELECT=$SELECT NUM_TARGETS=$NUM_TARGETS NUM_VICTIMS=$NUM_VICTIMS sh sel_dpp.sh"

source "$SOURCE_ROOT/sbatch/_job_common.sh"
