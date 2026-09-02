#!/bin/bash
# One matrix cell and one selector. submit_exact_amr_batch2.sh submits this
# template 40 times with MODEL, ATTACK, CLASS_PAIR, BUDGETS, and SELECT exported.
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=exact_amr_batch2
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs/exact_amr_batch2-%j.out

set -Eeuo pipefail

: "${MODEL:?MODEL was not exported by the submitter}"
: "${ATTACK:?ATTACK was not exported by the submitter}"
: "${CLASS_PAIR:?CLASS_PAIR was not exported by the submitter}"
: "${BUDGETS:?BUDGETS was not exported by the submitter}"
: "${SELECT:?SELECT was not exported by the submitter}"

case "$SELECT" in
    exact|a-mr) ;;
    *) echo "ERROR: batch-2 job supports SELECT=exact or SELECT=a-mr, got $SELECT" >&2; exit 1 ;;
esac

# Persistent paths on the cluster. The shared runtime stages code/data into the
# node-local SLURM_TMPDIR, then syncs results back under SOURCE_ROOT.
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
