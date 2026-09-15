#!/usr/bin/env bash
# attack_if path profile for the 10-target x 6-victim cross-architecture/K jobs.
# The experiment implementation is shared with the separate PoisonBase profile;
# this wrapper changes only the filesystem locations.

set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
ROOT="${ROOT:-$SOURCE_ROOT}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
DATA_ROOT="${DATA_ROOT:-/home/mmoslem3/scratch/data}"
RUN_ROOT="${RUN_ROOT:-${SLURM_TMPDIR:-}/attack_if}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/cross_arch_k_10x6_result}"

export SOURCE_ROOT ROOT ENV_ACTIVATE DATA_ROOT RUN_ROOT CACHE_ROOT RESULT_ROOT

COMMON="$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
[ -f "$COMMON" ] || {
    printf 'ERROR: shared runtime missing: %s\n' "$COMMON" >&2
    exit 1
}
# shellcheck disable=SC1090
source "$COMMON"
