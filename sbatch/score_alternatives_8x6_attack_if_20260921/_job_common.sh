#!/usr/bin/env bash
# Same experiment runtime, with the established attack_if filesystem layout.
set -Eeuo pipefail
export ROOT=/home/mmoslem3/scratch/attack_if
export ENV_ACTIVATE=/home/mmoslem3/ENV/bin/activate
export DATA_ROOT=/home/mmoslem3/scratch/data
export CACHE_ROOT="$ROOT/cache"
export RESULT_ROOT="$ROOT/score_alternatives_8x6_20260921_result"
: "${SLURM_TMPDIR:?Submit with sbatch; SLURM_TMPDIR is required}"
: "${SCORE_JOB_ID:?The individual job must set SCORE_JOB_ID}"
export RUN_ROOT="$SLURM_TMPDIR/attack_if_score_$SCORE_JOB_ID"
source "$ROOT/sbatch/score_alternatives_8x6_20260921/_job_common.sh"
