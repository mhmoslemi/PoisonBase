#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_tiny_gm_vgg13_5x5"
exec 9>"$BATCH/submission.lock"
flock -n 9 || { echo 'Another submitter is running.' >&2; exit 1; }
node "$ROOT/experiments/rebuttal_tiny_gm/submit.js" "$ROOT"
