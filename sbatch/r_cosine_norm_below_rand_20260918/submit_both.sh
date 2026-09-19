#!/usr/bin/env bash
# Submit 14 mirrored jobs from each checkout (28 Slurm jobs total).

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DRY_RUN="${DRY_RUN:-0}" bash "$SCRIPT_DIR/submit_poisonbase.sh"
DRY_RUN="${DRY_RUN:-0}" bash "$SCRIPT_DIR/submit_attack_if.sh"
