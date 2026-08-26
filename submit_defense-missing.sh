#!/bin/sh
# Submit only defense cells absent from the original run/retry sets.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/sbatch/missing_defense/submit.sh"
