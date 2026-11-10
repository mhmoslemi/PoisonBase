#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
flock -n "$ROOT/sbatch/rebuttal_tiny_gm_vgg13_5x5/submission.lock" \
    node "$ROOT/experiments/rebuttal_tiny_gm/submit_targets_vulcan.js" "$ROOT"
