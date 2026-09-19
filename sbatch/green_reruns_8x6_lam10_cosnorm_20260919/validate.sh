#!/usr/bin/env bash
# Block submission if any generated job differs from the requested protocol.

set -Eeuo pipefail
shopt -s nullglob

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
COMMON="$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
MANIFEST="$SCRIPT_DIR/manifest.tsv"
jobs=("$SCRIPT_DIR"/job_*.sh)

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

(( ${#jobs[@]} == 55 )) || die "expected 55 job files, found ${#jobs[@]}"
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
[[ "$manifest_rows" == 55 ]] || die "expected 55 manifest rows, found $manifest_rows"

for job in "${jobs[@]}"; do
    bash -n "$job"
    grep -Eq '^export XFULL_K=(1|3|10|20|30)$' "$job" || die "$job has invalid K"
    grep -qx 'export XFULL_NUM_TARGETS=8' "$job" || die "$job is not 8-target"
    grep -qx 'export XFULL_NUM_VICTIMS=6' "$job" || die "$job is not 6-victim"
    grep -qx 'export XFULL_COMPONENT=' "$job" || die "$job does not pin full BASIS"
    grep -qx 'export XFULL_BASE_DIST=cosine_norm' "$job" || die "$job has wrong distance"
    grep -qx 'export XFULL_LAMBDA_MARGIN=10' "$job" || die "$job has wrong margin"
    grep -qx 'export XFULL_VICTIM_EPOCHS=50' "$job" || die "$job has wrong victim epochs"
    grep -qx 'export XFULL_VICTIM_DECAY=40' "$job" || die "$job has wrong victim decay"
    grep -q '^export XFULL_RUN_NAME=.*_K\(1\|3\|10\|20\|30\)' "$job" || \
        die "$job run name does not identify K"
done

boyu=$(grep -h '^#SBATCH --account=aip-boyuwang$' "${jobs[@]}" | wc -l | tr -d ' ')
yiwei=$(grep -h '^#SBATCH --account=aip-yiweilu$' "${jobs[@]}" | wc -l | tr -d ' ')
[[ "$boyu" == 37 && "$yiwei" == 18 ]] || \
    die "wrong account split: boyuwang=$boyu yiweilu=$yiwei"

printf 'Validated: 55 green-cell jobs; 8 targets x 6 victims; accounts 37/18.\n'
