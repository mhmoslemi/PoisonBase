#!/usr/bin/env bash
# Merge four completed target partitions into one canonical FUS result.
set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/influence_fus_baselines_result}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -n "${IFB_RUN_NAME:-}" ] || die "IFB_RUN_NAME is unset"
[ -n "${IFB_MODEL:-}" ] || die "IFB_MODEL is unset"
[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
[ -f "$ROOT/merge_fus_parts.py" ] || die "missing $ROOT/merge_fus_parts.py"
target_file="$ROOT/target_sets/${IFB_MODEL}_gradmatch_dog-bird.json"
[ -s "$target_file" ] || die "missing target file: $target_file"

# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
python "$ROOT/merge_fus_parts.py" \
    --result-root "$RESULT_ROOT" \
    --run-name "$IFB_RUN_NAME" \
    --target-file "$target_file" \
    --model "$IFB_MODEL" \
    --class-pair dog-bird --parts 4 --targets 10 --victims 6
