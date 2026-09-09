#!/bin/sh
# Submit the same 30 coefficient-ablation jobs under the aip-yiweilu account.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export ACCOUNT=aip-yiweilu

exec sh "$SCRIPT_DIR/submit_distance_margin_coef_30.sh"
