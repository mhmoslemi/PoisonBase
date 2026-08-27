#!/usr/bin/env bash
# Submit the one still-blank defense.tex cell on Vulcan after its poison cache
# and 27/35 partial trials have been transferred from Killarney.

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
PYTHON_BIN="${PYTHON_BIN:-/home/mmoslem3/ENV/bin/python}"
JOB="$SOURCE_ROOT/sbatch/vulcan_remaining_20260827/defense_046_vgg13_sapa_b001_epic_ours_j_resume.sh"
LOG_ROOT="$SOURCE_ROOT/sbatch/logs2"
ATTACK_RUN='CIFAR10_VGG13BN_sapa_ours_dog-bird_b0.01_eps8_seed42_lam1_cosine_jacw1_worst0.05_ce5_tgt50'
DEFENSE_RUN="${ATTACK_RUN}__def-epic-s0.1-f2-d10"
TARGET_FILE='def_VGG13BN_sapa_dog-bird_b0.01_jacw1.json'

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

saved_trial_count() {
    "$PYTHON_BIN" - "$1" <<'PY'
import csv
import glob
import os
import sys

seen = set()
for path in glob.glob(os.path.join(sys.argv[1], "results*.csv")):
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                target = (row.get("target_idx") or "").strip()
                victim = (row.get("victim_id") or "").strip()
                if target and victim:
                    seen.add((target, victim))
    except OSError:
        pass
print(len(seen))
PY
}

[ -f "$JOB" ] || die "one-cell defense job is missing: $JOB"
[ "$(grep -c '^#SBATCH --cpus-per-task=1$' "$JOB" || true)" -eq 1 ] || die "job must request one CPU"
[ "$(grep -c '^#SBATCH --mem=7G$' "$JOB" || true)" -eq 1 ] || die "job must request 7 GB"
[ "$(grep -c '^#SBATCH --gpus-per-node=l40s:1$' "$JOB" || true)" -eq 1 ] || die "job must request one L40S"
[ "$(grep -c '^#SBATCH --mail' "$JOB" || true)" -eq 0 ] || die "mail directive found"
[ "$(grep -c '^source /home/mmoslem3/scratch/PoisonBase/sbatch/missing_tables_20260827/.*\.sh$' "$JOB" || true)" -eq 1 ] || \
    die "job must source exactly one defense cell"

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: sbatch --account=%s %s\n' "$ACCOUNT" "$JOB"
    exit 0
fi

case "$(hostname -s)" in
    vulcan*) ;;
    *) die "run this submitter on Vulcan (current host: $(hostname -s))" ;;
esac

[ -d "$SOURCE_ROOT/data/cifar-10-batches-py" ] || die "CIFAR-10 is missing under $SOURCE_ROOT/data"
[ -f "$SOURCE_ROOT/sbatch/_job_common.sh" ] || die "defense runtime is missing"
[ -x "$PYTHON_BIN" ] || die "Python environment is missing: $PYTHON_BIN"
[ -f "$SOURCE_ROOT/target_sets/$TARGET_FILE" ] || \
    die "transferred target file is missing: target_sets/$TARGET_FILE"

poison_cache="$SOURCE_ROOT/ours_result/$ATTACK_RUN/poison_cache"
deltas=$(find "$poison_cache" -maxdepth 1 -type f -name 'delta_*.pt' 2>/dev/null | wc -l | tr -d ' ')
[ "$deltas" -ge 7 ] || \
    die "only $deltas perturbations found; expected at least 7. Run the defense transfer first."

trials=$(saved_trial_count "$SOURCE_ROOT/defense_result/$DEFENSE_RUN")
[ "$trials" -ge 27 ] || \
    die "only $trials/35 partial trials found; expected at least 27. Run the defense transfer first."
if [ "$trials" -ge 35 ]; then
    printf 'Nothing submitted: defense cell is already complete (%s/35 trials).\n' "$trials"
    exit 0
fi

mkdir -p "$LOG_ROOT"
name=$(sed -n 's/^#SBATCH --job-name=//p' "$JOB")
active=$(squeue -h -u "${USER:-mmoslem3}" -n "$name" -o '%A' | head -n 1)
if [ -n "$active" ]; then
    printf 'Nothing submitted: %s is already active as job %s.\n' "$name" "$active"
    exit 0
fi

output=$(sbatch --parsable --account="$ACCOUNT" "$JOB")
job_id=${output%%;*}
case "$job_id" in *[!0-9]*|'') die "could not parse sbatch output: $output" ;; esac
printf 'submitted %s (%s/35 saved) -> %s\n' "$name" "$trials" "$job_id"
