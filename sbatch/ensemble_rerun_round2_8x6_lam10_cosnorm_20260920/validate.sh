#!/usr/bin/env bash
# Validate the requested second-round ensemble reruns before submission.

set -Eeuo pipefail
shopt -s nullglob

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
COMMON="$ROOT/sbatch/cross_arch_k_10x6_20260915/_job_common.sh"
MANIFEST="$SCRIPT_DIR/manifest.tsv"
jobs=("$SCRIPT_DIR"/job_*.sh)

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

(( ${#jobs[@]} == 34 )) || die "expected 34 job files, found ${#jobs[@]}"
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
[[ "$manifest_rows" == 34 ]] || die "expected 34 manifest rows, found $manifest_rows"

for job in "${jobs[@]}"; do
    bash -n "$job"
    grep -qx '#SBATCH --time=0-03:10:00' "$job" || die "$job has wrong walltime"
    grep -Eq '^export XFULL_K=(1|3|10|20|30)$' "$job" || die "$job has invalid K"
    grep -qx 'export XFULL_NUM_TARGETS=8' "$job" || die "$job is not 8-target"
    grep -qx 'export XFULL_NUM_VICTIMS=6' "$job" || die "$job is not 6-victim"
    grep -qx 'export XFULL_COMPONENT=' "$job" || die "$job does not pin full BASIS"
    grep -qx 'export XFULL_BASE_DIST=cosine_norm' "$job" || die "$job has wrong distance"
    grep -qx 'export XFULL_LAMBDA_MARGIN=10' "$job" || die "$job has wrong margin"
    grep -qx 'export XFULL_VICTIM_EPOCHS=50' "$job" || die "$job has wrong victim epochs"
    grep -qx 'export XFULL_VICTIM_DECAY=40' "$job" || die "$job has wrong victim decay"
    grep -qx 'export RESULT_ROOT="$ROOT/ensemble_rerun_round2_8x6_lam10_cosnorm_20260920_result"' "$job" || \
        die "$job does not use the fresh result root"
done

boyu=$(grep -h '^#SBATCH --account=aip-boyuwang$' "${jobs[@]}" | wc -l | tr -d ' ')
yiwei=$(grep -h '^#SBATCH --account=aip-yiweilu$' "${jobs[@]}" | wc -l | tr -d ' ')
[[ "$boyu" == 23 && "$yiwei" == 11 ]] || \
    die "wrong account split: boyuwang=$boyu yiweilu=$yiwei"

printf 'Validated: 34 jobs; 8 targets x 6 victims; accounts 23/11; walltime 3:10.\n'
