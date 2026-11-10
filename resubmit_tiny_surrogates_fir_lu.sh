#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
export TINY_REPLACE_OLD_JOBS=3197766,3197767,3197768,3197769
export TINY_REPLACE_ACCOUNT=def-yiweilu
exec sh "$ROOT/submit_tiny_gm_vgg13_fir_ghaddar.sh"
