#!/bin/bash
# One R-only/cosine-normalized rerun. The submitter supplies exactly one table
# configuration and overrides account, name, output path, and checkout root.
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=rcn_below_rand
#SBATCH --time=0-03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/rcn_below_rand-%j.out

set -Eeuo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT was not exported by the submitter}"
: "${PERSIST_DATA_ROOT:?PERSIST_DATA_ROOT was not exported by the submitter}"
: "${MODEL:?MODEL was not exported by the submitter}"
: "${ATTACK:?ATTACK was not exported by the submitter}"
: "${CLASS_PAIR:?CLASS_PAIR was not exported by the submitter}"
: "${BUDGETS:?BUDGETS was not exported by the submitter}"

export PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export JOB_KIND=attack
export SELECT=r
export BASE_DIST=cosine_norm
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export LAMBDA_MARGIN=10.0
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export NUM_TARGETS=8
export NUM_VICTIMS=5
export CRAFT_STEPS=250
export RECOMPUTE_DELTAS=0
export FORCE=0
export ORIGINAL_COMMAND="SOURCE_ROOT=$SOURCE_ROOT PERSIST_DATA_ROOT=$PERSIST_DATA_ROOT MODEL=$MODEL ATTACK=$ATTACK CLASS_PAIR=$CLASS_PAIR BUDGETS=$BUDGETS SELECT=r BASE_DIST=cosine_norm NUM_TARGETS=8 NUM_VICTIMS=5 CRAFT_STEPS=250 sh sel_dpp.sh"

source "$SOURCE_ROOT/sbatch/_job_common.sh"
