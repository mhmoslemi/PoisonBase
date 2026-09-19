#!/usr/bin/env bash
# Refuse submission unless every generated job is the intended K=20 protocol.

set -Eeuo pipefail
shopt -s nullglob

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
COMMON="$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
MANIFEST="$SCRIPT_DIR/manifest.tsv"
jobs=("$SCRIPT_DIR"/job_*.sh)

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

(( ${#jobs[@]} == 18 )) || die "expected 18 job files, found ${#jobs[@]}"
[[ -f "$MANIFEST" ]] || die "missing $MANIFEST"
[[ -f "$COMMON" ]] || die "missing $COMMON"

bash -n "$COMMON" "$SCRIPT_DIR/validate.sh" "$SCRIPT_DIR/submit_all.sh"
python3 - "$SCRIPT_DIR/generate_jobs.py" <<'PY'
import sys
path = sys.argv[1]
with open(path) as handle:
    compile(handle.read(), path, "exec")
PY

manifest_rows=$(awk 'NR > 1 && NF {count++} END {print count + 0}' "$MANIFEST")
[[ "$manifest_rows" == 18 ]] || die "expected 18 manifest rows, found $manifest_rows"

for job in "${jobs[@]}"; do
    bash -n "$job"
    grep -qx 'export XFULL_K=20' "$job" || die "$job does not set K=20"
    grep -qx 'export XFULL_NUM_TARGETS=10' "$job" || die "$job is not 10-target"
    grep -qx 'export XFULL_NUM_VICTIMS=6' "$job" || die "$job is not 6-victim"
    grep -qx 'export XFULL_BASE_DIST=cosine_norm' "$job" || die "$job has wrong distance"
    grep -qx 'export XFULL_LAMBDA_MARGIN=1' "$job" || die "$job has wrong lambda"
    grep -qx 'export XFULL_VICTIM_EPOCHS=70' "$job" || die "$job has wrong victim epochs"
    grep -qx 'export XFULL_VICTIM_DECAY=50' "$job" || die "$job has wrong victim decay"
    grep -q '^export XFULL_RUN_NAME=.*_K20' "$job" || die "$job has wrong run name"
    if grep -Eq '^export XFULL_K=(1|3|10|30)$' "$job"; then
        die "$job contains a forbidden lower/non-reference K"
    fi
done

boyu=$(grep -h '^#SBATCH --account=aip-boyuwang$' "${jobs[@]}" | wc -l | tr -d ' ')
yiwei=$(grep -h '^#SBATCH --account=aip-yiweilu$' "${jobs[@]}" | wc -l | tr -d ' ')
[[ "$boyu" == 12 && "$yiwei" == 6 ]] || \
    die "wrong account split: boyuwang=$boyu yiweilu=$yiwei"

printf 'Validated: 18 K=20-only jobs; 10 targets x 6 victims; accounts 12/6.\n'
