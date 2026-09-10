#!/bin/sh
# Submit the same four BP diagnostic jobs on aip-boyuwang.

set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
export ACCOUNT=aip-boyuwang
exec sh "$SCRIPT_DIR/submit_bp_failure_diagnostic_4.sh"
