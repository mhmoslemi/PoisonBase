#!/bin/bash
# One BASIS rerun with cosine-normalized distance and longer victim training.
# submit_all.sh supplies exactly one configuration per Slurm job.
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=bm70_rerun
#SBATCH --time=0-03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/bm70_rerun-%j.out

set -Eeuo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT was not exported by the submitter}"
: "${PERSIST_DATA_ROOT:?PERSIST_DATA_ROOT was not exported by the submitter}"
: "${MODEL:?MODEL was not exported by the submitter}"
: "${ATTACK:?ATTACK was not exported by the submitter}"
: "${CLASS_PAIR:?CLASS_PAIR was not exported by the submitter}"
: "${BUDGETS:?BUDGETS was not exported by the submitter}"
: "${LAMBDA_MARGIN:?LAMBDA_MARGIN was not exported by the submitter}"

case "$LAMBDA_MARGIN" in
    1|100) ;;
    *) echo "ERROR: expected LAMBDA_MARGIN=1 or 100, got $LAMBDA_MARGIN" >&2; exit 1 ;;
esac

unset DISTANCE_MARGIN_COEF
export PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export JOB_KIND=attack
export SELECT=ours
export BASE_DIST=cosine_norm
export USE_JACOBIAN_SCORE=0
export JACOBIAN_WEIGHT=1.0
export JACOBIAN_BATCH_SIZE=64
export SHARP_MODE=worst
export SHARP_SIGMA=0.05
export TARGET_SELECT=''
export NUM_TARGETS=8
export NUM_VICTIMS=5
export CRAFT_STEPS=250
export VICTIM_EPOCHS=70
export VICTIM_DECAY=50
export RECOMPUTE_DELTAS=0
export FORCE=0
export ORIGINAL_COMMAND="SOURCE_ROOT=$SOURCE_ROOT PERSIST_DATA_ROOT=$PERSIST_DATA_ROOT MODEL=$MODEL ATTACK=$ATTACK CLASS_PAIR=$CLASS_PAIR BUDGETS=$BUDGETS SELECT=ours BASE_DIST=cosine_norm LAMBDA_MARGIN=$LAMBDA_MARGIN USE_JACOBIAN_SCORE=0 NUM_TARGETS=8 NUM_VICTIMS=5 CRAFT_STEPS=250 VICTIM_EPOCHS=70 VICTIM_DECAY=50 sh sel_dpp.sh"

source "$SOURCE_ROOT/sbatch/_job_common.sh"
