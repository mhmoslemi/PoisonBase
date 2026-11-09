#!/bin/sh
# Run from the Fir checkout, normally ~/scratch/PoisonBase.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_tiny_gm_vgg13_5x5"
exec 9>"$BATCH/submission_fir.lock"
flock -n 9 || { echo 'Another Fir submitter is running.' >&2; exit 1; }
if ! command -v node >/dev/null; then
    NODE_MODULE="$(module -t spider nodejs 2>&1 | awk '/^nodejs\/[0-9]/{print $1; exit}')"
    [ -n "$NODE_MODULE" ] || { echo 'ERROR: node is unavailable; module spider nodejs' >&2; exit 1; }
    module load "$NODE_MODULE"
fi
node "$ROOT/experiments/rebuttal_tiny_gm/submit.js" "$ROOT" fir
