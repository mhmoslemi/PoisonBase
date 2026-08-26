#!/bin/sh
# The defense retries require a missing attack cache, so submit the full chain.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/submit_remaining.sh"
