#!/usr/bin/env bash
# PoisonBase-path wrapper for the 54 jobs moved from Yiwei to Boyu.

set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export ROOT ENV_ACTIVATE DATA_ROOT

COMMON="$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
[ -f "$COMMON" ] || {
    printf 'ERROR: shared runtime missing: %s\n' "$COMMON" >&2
    exit 1
}
# shellcheck disable=SC1090
source "$COMMON"
