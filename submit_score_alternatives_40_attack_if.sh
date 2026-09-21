#!/usr/bin/env bash
# One command submits all 40 new scoring experiments with attack_if paths.
set -Eeuo pipefail
SCRIPT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec bash "$SCRIPT_ROOT/sbatch/score_alternatives_8x6_attack_if_20260921/submit_all.sh" "$@"
