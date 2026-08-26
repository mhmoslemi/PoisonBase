#!/bin/sh
# Submit every one-cell defense job generated under sbatch/defense/.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/sbatch/submit_defense.sh"
