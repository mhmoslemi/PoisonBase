#!/bin/sh
# All 62 generated defenses currently have completed matching attack cells.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/sbatch/submit_defense.sh"
