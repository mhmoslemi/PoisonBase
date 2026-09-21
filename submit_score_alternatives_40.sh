#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec bash "$ROOT/sbatch/score_alternatives_8x6_20260921/submit_all.sh" "$@"
