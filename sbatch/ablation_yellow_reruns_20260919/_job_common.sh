#!/usr/bin/env bash
# Shared settings for the nine yellow-highlighted abaltion_diff.tex reruns.
set -Eeuo pipefail

: "${MODEL:?MODEL must be set by the job file}"
: "${ATTACK:?ATTACK must be set by the job file}"
: "${BUDGETS:?BUDGETS must be set by the job file}"
: "${SELECT:?SELECT must be set by the job file}"

export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
export PYTHON_ENV="${PYTHON_ENV:-/home/mmoslem3/ENV}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-$PYTHON_ENV/bin/activate}"
export CLASS_PAIR=dog-bird
export BASE_DIST=cosine_norm

case "$SELECT" in
    r|minus-m|exact)
        # These component selectors use the original table protocol. Their
        # selection formulas do not use the BASIS margin weight; cosine_norm
        # gives the reruns a fresh result identity without replacing old data.
        source "$SOURCE_ROOT/sbatch/bench_missing_components/_job_common.sh"
        ;;
    ours)
        # Retain the intended anomaly-rerun setup that previously failed before
        # Python started: lambda=1, normalized cosine, and 70-epoch victims.
        export LAMBDA_MARGIN=1
        source "$SOURCE_ROOT/sbatch/basis_margin_reruns_v70_20260919/job.sh"
        ;;
    *)
        echo "ERROR: unsupported yellow-cell selector: $SELECT" >&2
        exit 1
        ;;
esac
