#!/usr/bin/env bash
# Submit the unfinished augment-extra.tex cells on Vulcan, shortest first.
# Missing or target-pairing poison optimizations are submitted first and attached as row-level
# afterok dependencies.  Re-running this file skips completed/active work.

# Users commonly launch these submitters with `sh file.sh`; on Vulcan that is
# dash, which cannot provide the arrays used below.  Relaunch under Bash while
# preserving arguments and environment (including DRY_RUN).
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"

set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ATTACK_DIR="$ROOT/sbatch/augment_extra_attack"
EVAL_DIR="$ROOT/sbatch/augment_extra"
DRY_RUN="${DRY_RUN:-0}"

[ -d "$ATTACK_DIR" ] || { echo "missing $ATTACK_DIR" >&2; exit 1; }
[ -d "$EVAL_DIR" ] || { echo "missing $EVAL_DIR" >&2; exit 1; }
[ "$(find "$ATTACK_DIR" -maxdepth 1 -type f -name '*.sh' | wc -l)" -eq 11 ] || {
    echo 'expected exactly 11 poison-prerequisite SBATCH files' >&2; exit 1; }
[ "$(find "$EVAL_DIR" -maxdepth 1 -type f -name '*.sh' | wc -l)" -eq 248 ] || {
    echo 'expected exactly 248 unfinished one-cell evaluation SBATCH files' >&2; exit 1; }

mkdir -p "$ROOT/sbatch/logs"
source /home/mmoslem3/ENV/bin/activate
cd "$ROOT"

# Refuse to launch if a missing poison cache is not covered by one of the
# generated prerequisite jobs.  Rows waiting on a prerequisite are checked
# again inside each evaluation job after its afterok dependency completes.
python - "$ROOT" "$EVAL_DIR" "$ATTACK_DIR" <<'PY'
import glob, os, re, sys
root, eval_dir, attack_dir = sys.argv[1:]
sys.path.insert(0, root)
import defense

def exports(path):
    out = {}
    for line in open(path):
        m = re.match(r"export ([A-Z0-9_]+)=(?:'([^']*)'|([^\s]+))", line)
        if m:
            out[m.group(1)] = m.group(2) or m.group(3)
    return out

prereqs = {exports(p).get('EXPECTED_RUN_NAME')
           for p in glob.glob(os.path.join(attack_dir, '*.sh'))}
rows = {}
for path in glob.glob(os.path.join(eval_dir, '*.sh')):
    e = exports(path)
    if e.get('ROW_ID'):
        rows[e['ROW_ID']] = e

unexpected, bad_intersection, ready, deferred = [], [], 0, 0
keys = ('RUN_RANDOM', 'RUN_GREEDY', 'RUN_DPP2', 'RUN_DPP025', 'RUN_DPP01')
for row, e in sorted(rows.items()):
    sets, missing = [], []
    for key in keys:
        run = e[key]
        path = os.path.join(root, 'ours_result', run)
        targets = set(defense.cached_targets(path)) if os.path.isdir(path) else set()
        sets.append(targets)
        if len(targets) < 5:
            missing.append((run, len(targets)))
            if run not in prereqs:
                unexpected.append((row, run, len(targets)))
    if missing:
        deferred += 1
    else:
        have = set.intersection(*sets)
        if len(have) < 5:
            bad_intersection.append((row, len(have)))
        else:
            ready += 1
if unexpected:
    for row, run, count in unexpected:
        print('UNEXPECTED missing cache: row %s %s (%d/5)' % (row, run, count),
              file=sys.stderr)
    raise SystemExit('preflight failed: missing poison caches without prerequisite jobs')
if bad_intersection:
    for row, count in bad_intersection:
        print('BAD target intersection: row %s has %d/5' % (row, count),
              file=sys.stderr)
    raise SystemExit('preflight failed: selectors are not paired on five targets')
print('preflight: %d rows ready, %d rows covered by poison prerequisites'
      % (ready, deferred))
PY

declare -A ACTIVE_BY_NAME COMPLETE_RUN ROW_DEPS
while IFS='|' read -r jid jname; do
    [ -n "$jname" ] && ACTIVE_BY_NAME["$jname"]="$jid"
done < <(squeue -h -u "$USER" -o '%i|%j')

while IFS= read -r run; do
    [ -n "$run" ] && COMPLETE_RUN["$run"]=1
done < <(python - "$ROOT/augment_extra_result" <<'PY'
import json, os, sys
root = sys.argv[1]
if os.path.isdir(root):
    for name in os.listdir(root):
        path = os.path.join(root, name, 'summary.json')
        try:
            with open(path) as handle:
                s = json.load(handle)
            if int(s.get('num_targets') or 0) == 5 and int(s.get('num_trials') or 0) == 20:
                print(name)
        except (OSError, ValueError, TypeError):
            pass
PY
)

job_value() {
    local file="$1" key="$2"
    sed -n "s/^export ${key}=//p" "$file" | tail -1 | sed "s/^'//;s/'$//"
}

job_name() {
    sed -n 's/^#SBATCH --job-name=//p' "$1" | head -1
}

cached_targets() {
    python - "$ROOT/ours_result/$1" <<'PY'
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(sys.argv[1])))
import defense
path = sys.argv[1]
print(len(defense.cached_targets(path)) if os.path.isdir(path) else 0)
PY
}

add_row_dependency() {
    local row="$1" jid="$2"
    if [ -n "${ROW_DEPS[$row]:-}" ]; then
        ROW_DEPS[$row]="${ROW_DEPS[$row]}:$jid"
    else
        ROW_DEPS[$row]="$jid"
    fi
}

submit_one() {
    local file="$1" dep="${2:-}" out
    if [ "$DRY_RUN" = 1 ]; then
        if [ -n "$dep" ]; then
            echo "DRY_RUN sbatch --dependency=afterok:$dep $file" >&2
        else
            echo "DRY_RUN sbatch $file" >&2
        fi
        printf 'DRY%05d\n' "$((++DRY_COUNTER))"
        return
    fi
    if [ -n "$dep" ]; then
        out="$(sbatch --dependency="afterok:$dep" "$file")"
    else
        out="$(sbatch "$file")"
    fi
    echo "$out" >&2
    awk '{print $NF}' <<<"$out"
}

DRY_COUNTER=0
echo '=== poison prerequisites (shortest first) ==='
while IFS='|' read -r _seconds file; do
    run="$(job_value "$file" EXPECTED_RUN_NAME)"
    row="$(job_value "$file" PREREQ_ROW)"
    required="$(job_value "$file" REQUIRED_CACHED_TARGETS)"
    required="${required:-5}"
    name="$(job_name "$file")"
    count="$(cached_targets "$run")"
    if [ "$count" -ge "$required" ]; then
        echo "ready: row $row $run ($count/$required cached targets)"
        continue
    fi
    if [ -n "${ACTIVE_BY_NAME[$name]:-}" ]; then
        jid="${ACTIVE_BY_NAME[$name]}"
        echo "active: $name job $jid ($count/$required cached targets)"
    else
        jid="$(submit_one "$file")"
        echo "queued prerequisite: $name job $jid ($count/$required cached targets)"
        ACTIVE_BY_NAME["$name"]="$jid"
    fi
    add_row_dependency "$row" "$jid"
done < <(
    for file in "$ATTACK_DIR"/*.sh; do
        t="$(sed -n 's/^#SBATCH --time=//p' "$file")"
        IFS='-:' read -r d h m s <<<"$t"
        printf '%09d|%s\n' "$((10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s))" "$file"
    done | sort -t'|' -k1,1n -k2,2
)

echo '=== unfinished augmentation cells (shortest first) ==='
submitted=0
skipped_complete=0
skipped_active=0
while IFS='|' read -r _seconds file; do
    name="$(job_name "$file")"
    row="$(job_value "$file" ROW_ID)"
    run="$(job_value "$file" EXPECTED_DEFENSE_RUN)"
    if [ -n "${COMPLETE_RUN[$run]:-}" ]; then
        echo "complete: $name"
        skipped_complete=$((skipped_complete + 1))
        continue
    fi
    if [ -n "${ACTIVE_BY_NAME[$name]:-}" ]; then
        echo "active: $name job ${ACTIVE_BY_NAME[$name]}"
        skipped_active=$((skipped_active + 1))
        continue
    fi
    dep="${ROW_DEPS[$row]:-}"
    jid="$(submit_one "$file" "$dep")"
    ACTIVE_BY_NAME["$name"]="$jid"
    submitted=$((submitted + 1))
done < <(
    for file in "$EVAL_DIR"/*.sh; do
        t="$(sed -n 's/^#SBATCH --time=//p' "$file")"
        IFS='-:' read -r d h m s <<<"$t"
        printf '%09d|%s\n' "$((10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s))" "$file"
    done | sort -t'|' -k1,1n -k2,2
)

echo "done: submitted=$submitted complete=$skipped_complete active=$skipped_active"
echo 'All generated SBATCH files contain exactly one unfinished table cell.'
