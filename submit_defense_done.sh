#!/bin/sh
# Sixty defenses are complete. The remaining two require one missing attack
# cache, so submit that dependency chain instead of resubmitting all defenses.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/submit_remaining.sh"
