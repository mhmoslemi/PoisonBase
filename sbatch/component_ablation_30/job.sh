#!/bin/bash
# One matrix cell and one component selector. The root submitter launches this
# template 180 times and overrides the account/job name/output for every job.
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=component_ablation_30
#SBATCH --time=0-03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/component_ablation_30-%j.out

set -Eeuo pipefail

: "${MODEL:?MODEL was not exported by the submitter}"
: "${ATTACK:?ATTACK was not exported by the submitter}"
: "${CLASS_PAIR:?CLASS_PAIR was not exported by the submitter}"
: "${BUDGETS:?BUDGETS was not exported by the submitter}"
: "${SELECT:?SELECT was not exported by the submitter}"

case "$SELECT" in
    minus-m|r|a|a-minus-m|a-plus-r|minus-m-times-r) ;;
    *) echo "ERROR: unsupported component SELECT=$SELECT" >&2; exit 1 ;;
esac

# Persistent cluster paths. The shared runtime stages both under SLURM_TMPDIR
# and syncs the run directory back to this exact PoisonBase tree on exit.
POISON_ROOT=/home/mmoslem3/scratch/PoisonBase
POISON_DATA_ROOT="$POISON_ROOT/data"
export SOURCE_ROOT="$POISON_ROOT"
export PERSIST_DATA_ROOT="$POISON_DATA_ROOT"

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
export ORIGINAL_COMMAND="SOURCE_ROOT=$SOURCE_ROOT PERSIST_DATA_ROOT=$PERSIST_DATA_ROOT MODEL=$MODEL ATTACK=$ATTACK CLASS_PAIR=$CLASS_PAIR BUDGETS=$BUDGETS SELECT=$SELECT NUM_TARGETS=$NUM_TARGETS NUM_VICTIMS=$NUM_VICTIMS sh sel_dpp.sh"

source "$SOURCE_ROOT/sbatch/_job_common.sh"
