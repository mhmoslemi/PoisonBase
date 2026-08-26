#!/bin/sh
# Submit every one-cell attack job generated under sbatch/attack/.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/sbatch/submit_attack.sh"
