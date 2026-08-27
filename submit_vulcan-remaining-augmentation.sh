#!/usr/bin/env bash
# Submit only the 21 RandAugment cells still blank in augment-extra.tex.
# Run on Vulcan after transfer_augmentation_artifacts_to_vulcan.sh.

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
PYTHON_BIN="${PYTHON_BIN:-/home/mmoslem3/ENV/bin/python}"
JOB_ROOT="$SOURCE_ROOT/sbatch/vulcan_remaining_20260827"
LOG_ROOT="$SOURCE_ROOT/sbatch/logs2"
EXPECTED_JOBS=21

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

augmentation_jobs() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name 'augment_*_resume.sh' -print
}

line_count() {
    grep -c "$1" "$2" 2>/dev/null || true
}

[ -d "$JOB_ROOT" ] || die "job directory missing: $JOB_ROOT"
count=$(augmentation_jobs | wc -l | tr -d ' ')
[ "$count" -eq "$EXPECTED_JOBS" ] || \
    die "expected $EXPECTED_JOBS one-cell augmentation jobs; found $count"

names=$(mktemp)
manifest=$(mktemp)
trap 'unlink "$names" "$manifest" 2>/dev/null || true' EXIT HUP INT TERM

augmentation_jobs |
while IFS= read -r job; do
    [ "$(line_count '^#SBATCH --job-name=' "$job")" -eq 1 ] || die "expected one job name: $job"
    [ "$(line_count '^#SBATCH --time=' "$job")" -eq 1 ] || die "expected one walltime: $job"
    [ "$(line_count '^#SBATCH --cpus-per-task=1$' "$job")" -eq 1 ] || die "job must request one CPU: $job"
    [ "$(line_count '^#SBATCH --mem=7G$' "$job")" -eq 1 ] || die "job must request 7 GB: $job"
    [ "$(line_count '^#SBATCH --gpus-per-node=l40s:1$' "$job")" -eq 1 ] || die "job must request one L40S: $job"
    [ "$(line_count '^#SBATCH --mail' "$job")" -eq 0 ] || die "mail directive found: $job"
    [ "$(line_count '^#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/' "$job")" -eq 1 ] || \
        die "output is not routed to logs2: $job"
    [ "$(line_count '^export EXPECTED_SAVED_TRIALS=[0-9][0-9]*$' "$job")" -eq 1 ] || \
        die "saved-trial checkpoint is missing: $job"
    [ "$(line_count '^source /home/mmoslem3/scratch/PoisonBase/sbatch/augment_extra/.*\.sh$' "$job")" -eq 1 ] || \
        die "job must source exactly one original one-cell template: $job"
    requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
    seconds=$(printf '%s\n' "$requested" | awk -F '[-:]' \
        '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }')
    [ "$seconds" -le 25200 ] || die "augmentation job exceeds seven hours: $job"
    sed -n 's/^#SBATCH --job-name=//p' "$job" >> "$names"
done

[ "$(sort -u "$names" | wc -l | tr -d ' ')" -eq "$EXPECTED_JOBS" ] || die "duplicate Slurm job names"

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: %s one-cell augmentation jobs validated.\n' "$EXPECTED_JOBS"
    augmentation_jobs |
    while IFS= read -r job; do
        requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
        seconds=$(printf '%s\n' "$requested" | awk -F '[-:]' \
            '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }')
        printf '%010d\t%s\n' "$seconds" "$job"
    done | sort -n -k1,1 -k2,2 | cut -f2- |
    while IFS= read -r job; do
        printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"
    done
    exit 0
fi

case "$(hostname -s)" in
    vulcan*) ;;
    *) die "run this submitter on Vulcan (current host: $(hostname -s))" ;;
esac

[ -d "$SOURCE_ROOT/data/cifar-10-batches-py" ] || die "CIFAR-10 is missing under $SOURCE_ROOT/data"
[ -f "$SOURCE_ROOT/sbatch/_augment_extra_job_common.sh" ] || die "augmentation runtime is missing"
[ -x "$PYTHON_BIN" ] || die "Python environment is missing: $PYTHON_BIN"

# Validate the transferred partial trials and all selector poison caches.
"$PYTHON_BIN" - "$SOURCE_ROOT" "$JOB_ROOT" > "$manifest" <<'PY'
import csv
import glob
import os
import re
import sys

root, job_root = sys.argv[1:]
jobs = sorted(glob.glob(os.path.join(job_root, "augment_*_resume.sh")))
poison_runs = set()
records = []

def read_exports(path):
    values = {}
    with open(path) as handle:
        for line in handle:
            match = re.match(r"export ([A-Z0-9_]+)=(?:'([^']*)'|([^\s]+))", line)
            if match:
                values[match.group(1)] = match.group(2) or match.group(3)
    return values

def trial_count(path):
    seen = set()
    for csv_path in glob.glob(os.path.join(path, "results*.csv")):
        try:
            with open(csv_path, newline="") as handle:
                for row in csv.DictReader(handle):
                    target = (row.get("target_idx") or "").strip()
                    victim = (row.get("victim_id") or "").strip()
                    if target and victim:
                        seen.add((target, victim))
        except OSError:
            pass
    return len(seen)

for job in jobs:
    text = open(job).read()
    source = re.search(r"^source .*/sbatch/augment_extra/([^/\s]+\.sh)$", text, re.MULTILINE)
    expected = re.search(r"^export EXPECTED_SAVED_TRIALS=(\d+)$", text, re.MULTILINE)
    name = re.search(r"^#SBATCH --job-name=(\S+)$", text, re.MULTILINE)
    wall = re.search(r"^#SBATCH --time=(\d+)-(\d+):(\d+):(\d+)$", text, re.MULTILINE)
    if not all((source, expected, name, wall)):
        raise SystemExit("incomplete wrapper metadata: %s" % job)

    template = os.path.join(root, "sbatch", "augment_extra", source.group(1))
    if not os.path.isfile(template):
        raise SystemExit("missing source template: %s" % template)
    exports = read_exports(template)
    for key in ("RUN_RANDOM", "RUN_GREEDY", "RUN_DPP2", "RUN_DPP025", "RUN_DPP01"):
        run = exports.get(key)
        if not run:
            raise SystemExit("%s missing from %s" % (key, template))
        poison_runs.add(run)

    result = exports.get("EXPECTED_DEFENSE_RUN")
    if not result:
        raise SystemExit("EXPECTED_DEFENSE_RUN missing from %s" % template)
    actual = trial_count(os.path.join(root, "augment_extra_result", result))
    minimum = int(expected.group(1))
    if actual < minimum:
        raise SystemExit(
            "%s has only %d saved trials; expected at least %d. "
            "Run the augmentation transfer first." % (result, actual, minimum))
    days, hours, minutes, seconds = map(int, wall.groups())
    requested = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    records.append((requested, job, name.group(1), actual))

if len(poison_runs) != 25:
    raise SystemExit("expected 25 selector poison caches; found %d" % len(poison_runs))
for run in sorted(poison_runs):
    cache = os.path.join(root, "ours_result", run, "poison_cache")
    deltas = glob.glob(os.path.join(cache, "delta_*.pt"))
    if len(deltas) < 5:
        raise SystemExit(
            "%s has only %d perturbations; expected at least 5. "
            "Run the augmentation transfer first." % (run, len(deltas)))

for requested, job, name, actual in sorted(records):
    print("%d\t%s\t%s\t%d" % (requested, job, name, actual))
PY

printf 'Verified 21 partial cells and all 25 selector poison caches.\n'
mkdir -p "$LOG_ROOT"
squeue -h -u "${USER:-mmoslem3}" -o '%.200j' > "$names"

while IFS="$(printf '\t')" read -r _seconds job name trials; do
    if [ "$trials" -ge 20 ]; then
        printf 'skip complete: %s (%s/20 trials)\n' "$name" "$trials"
        continue
    fi
    if grep -Fqx "$name" "$names"; then
        printf 'skip active duplicate: %s\n' "$name"
        continue
    fi
    output=$(sbatch --parsable --account="$ACCOUNT" "$job")
    job_id=${output%%;*}
    case "$job_id" in *[!0-9]*|'') die "could not parse sbatch output: $output" ;; esac
    printf 'submitted %s (%s/20 saved) -> %s\n' "$name" "$trials" "$job_id"
    printf '%s\n' "$name" >> "$names"
done < "$manifest"

printf 'Augmentation submission pass finished; output will appear in sbatch/logs2.\n'
