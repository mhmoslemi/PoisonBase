#!/bin/sh
# Submit the PoisonBase-rooted 36-job margin-weight sweep on aip-boyuwang.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
export ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
export PERSIST_DATA_ROOT="${PERSIST_DATA_ROOT:-$SOURCE_ROOT/data}"
export ACCOUNT=aip-boyuwang

exec sh "$SCRIPT_DIR/submit_margin_weight_36_yiweilu.sh"
