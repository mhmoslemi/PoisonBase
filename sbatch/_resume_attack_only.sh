#!/usr/bin/env bash
# Guard for attack jobs that are allowed to resume an existing partial run, but
# must never silently start that table cell again from zero.

set -Eeuo pipefail

resume_trial_count() {
    local run_dir="$1"
    local resume_python="${PYTHON_ENV:-/home/mmoslem3/ENV}/bin/python"
    if [ ! -x "$resume_python" ]; then
        resume_python="$(command -v python3 || true)"
    fi
    [ -n "$resume_python" ] || {
        printf 'ERROR: Python 3 is required to inspect resume CSV files\n' >&2
        return 1
    }
    "$resume_python" - "$run_dir" <<'PY'
import csv
import glob
import os
import sys

run_dir = sys.argv[1]
paths = [os.path.join(run_dir, 'results.csv')]
paths.extend(sorted(glob.glob(os.path.join(run_dir, 'results_rank*.csv'))))
completed = set()
for path in paths:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        continue
    try:
        with open(path, newline='') as handle:
            for row in csv.DictReader(handle):
                target = row.get('target_idx')
                victim = row.get('victim_id')
                if target not in (None, '') and victim not in (None, ''):
                    completed.add((int(target), int(victim)))
    except (OSError, csv.Error, TypeError, ValueError) as exc:
        raise SystemExit('cannot read resume state from %s: %s' % (path, exc))
print(len(completed))
PY
}

if [ "${1:-}" = --count ]; then
    [ "$#" -eq 2 ] || {
        printf 'usage: %s --count RUN_DIR\n' "$0" >&2
        exit 2
    }
    resume_trial_count "$2"
    exit 0
fi

[ "${RESUME_ONLY:-0}" = 1 ] || {
    printf 'ERROR: resume guard requires RESUME_ONLY=1\n' >&2
    exit 1
}
[ -n "${RESUME_RUN_NAME:-}" ] || {
    printf 'ERROR: RESUME_RUN_NAME is unset\n' >&2
    exit 1
}

resume_source_root="${SOURCE_ROOT:-/home/mmoslem3/scratch/attack_if}"
resume_run_dir="$resume_source_root/ours_result/$RESUME_RUN_NAME"
resume_total="${RESUME_TOTAL_TRIALS:-48}"
resume_minimum="${RESUME_MIN_COMPLETED:-1}"

case "$resume_total:$resume_minimum" in
    *[!0-9:]*|:*|*:) printf 'ERROR: invalid resume trial counts\n' >&2; exit 1 ;;
esac
[ "$resume_total" -gt 0 ] || {
    printf 'ERROR: RESUME_TOTAL_TRIALS must be positive\n' >&2
    exit 1
}
[ "$resume_minimum" -lt "$resume_total" ] || {
    printf 'ERROR: RESUME_MIN_COMPLETED must be below RESUME_TOTAL_TRIALS\n' >&2
    exit 1
}
[ -d "$resume_run_dir" ] || {
    printf 'ERROR: resume-only run directory is missing: %s\n' "$resume_run_dir" >&2
    exit 1
}

resume_completed="$(resume_trial_count "$resume_run_dir")"
if [ "$resume_completed" -lt "$resume_minimum" ]; then
    printf 'ERROR: resume-only job expected at least %s/%s completed trials, found %s in %s\n' \
        "$resume_minimum" "$resume_total" "$resume_completed" "$resume_run_dir" >&2
    exit 1
fi
if [ "$resume_completed" -ge "$resume_total" ]; then
    printf 'resume-only: %s/%s trials are already complete; nothing to submit for this cell\n' \
        "$resume_completed" "$resume_total"
    exit 0
fi

printf 'resume-only: verified %s/%s completed trials in %s; final_update.py will run only the remaining %s\n' \
    "$resume_completed" "$resume_total" "$resume_run_dir" \
    "$((resume_total - resume_completed))"
