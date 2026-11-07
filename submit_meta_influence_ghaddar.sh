#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
export SUBMIT_ACCOUNT=aip-bghaddar
exec sh "$ROOT/submit_meta_influence_lu.sh" "$@"
