#!/bin/sh
# Submit only the work still required after reconciling the current logs.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/sbatch/remaining/submit.sh"
