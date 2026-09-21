#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export TINYIMAGENET_SLURM_ACCOUNT=aip-yiweilu
exec bash "$SCRIPT_ROOT/sbatch/tinyimagenet20_gm_8x6_20260921/submit_all.sh" "$@"
