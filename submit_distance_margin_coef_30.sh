#!/bin/sh
# Validate and submit the 30 ConvNet distance/margin coefficient-ablation jobs.
# Use ROOT=/path/to/checkout DRY_RUN=1 sh submit_distance_margin_coef_30.sh locally.

set -eu

ROOT="${ROOT:-/home/mmoslem3/scratch/attack_if}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
JOB_ROOT="$ROOT/sbatch/distance_margin_coef_30"
LOG_ROOT="$ROOT/sbatch/logs"
EXPECTED_JOBS=30
JOB_LIST=""
JOB_NAMES=""
MATRIX=""
ACTIVE_NAMES=""

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
    for path in "$JOB_LIST" "$JOB_NAMES" "$MATRIX" "$ACTIVE_NAMES"; do
        [ -z "$path" ] || [ ! -e "$path" ] || unlink "$path"
    done
}
trap cleanup EXIT HUP INT TERM

line_count() {
    pattern="$1"
    file="$2"
    count=$(grep -c "$pattern" "$file" 2>/dev/null || true)
    printf '%s\n' "$count"
}

JOB_LIST=$(mktemp)
JOB_NAMES=$(mktemp)
MATRIX=$(mktemp)

[ -d "$JOB_ROOT" ] || die "job directory missing: $JOB_ROOT"
[ -f "$JOB_ROOT/_job.sh" ] || die "shared job runtime missing: $JOB_ROOT/_job.sh"
[ "$(line_count '^export MODEL=ConvNetBN$' "$JOB_ROOT/_job.sh")" -eq 1 ] ||
    die "shared runtime is not fixed to ConvNetBN"
[ "$(line_count '^export CLASS_PAIR=dog-bird$' "$JOB_ROOT/_job.sh")" -eq 1 ] ||
    die "shared runtime is not fixed to dog-bird"
[ "$(line_count '^export SELECT=ours$' "$JOB_ROOT/_job.sh")" -eq 1 ] ||
    die "shared runtime is not fixed to SELECT=ours"
[ "$(line_count '^export USE_JACOBIAN_SCORE=0$' "$JOB_ROOT/_job.sh")" -eq 1 ] ||
    die "shared runtime does not disable the Jacobian score"
[ "$(line_count '^export NUM_TARGETS=6$' "$JOB_ROOT/_job.sh")" -eq 1 ] ||
    die "shared runtime does not use six targets"
[ "$(line_count '^export NUM_VICTIMS=6$' "$JOB_ROOT/_job.sh")" -eq 1 ] ||
    die "shared runtime does not use six victims"
find "$JOB_ROOT" -maxdepth 1 -type f -name 'dmcoef_*.sh' -print | sort > "$JOB_LIST"

count=$(wc -l < "$JOB_LIST" | tr -d ' ')
[ "$count" -eq "$EXPECTED_JOBS" ] ||
    die "expected $EXPECTED_JOBS one-cell jobs; found $count"

while IFS= read -r job; do
    [ "$(line_count '^#SBATCH --account=aip-boyuwang$' "$job")" -eq 1 ] ||
        die "job does not use the aip-boyuwang account: $job"
    [ "$(line_count '^#SBATCH --job-name=' "$job")" -eq 1 ] ||
        die "expected one Slurm job name in $job"
    [ "$(line_count '^#SBATCH --time=0-02:40:00$' "$job")" -eq 1 ] ||
        die "job does not request exactly 2h40m: $job"
    [ "$(line_count '^#SBATCH --cpus-per-task=1$' "$job")" -eq 1 ] ||
        die "job does not request exactly one CPU: $job"
    [ "$(line_count '^#SBATCH --mem=7G$' "$job")" -eq 1 ] ||
        die "job does not request 7 GB: $job"
    [ "$(line_count '^#SBATCH --gres=gpu:l40s:1$' "$job")" -eq 1 ] ||
        die "job does not request one L40S: $job"
    [ "$(line_count '^#SBATCH --mail' "$job")" -eq 0 ] ||
        die "email directive found in $job"
    [ "$(line_count '^export ATTACK=' "$job")" -eq 1 ] ||
        die "job must export one attack: $job"
    [ "$(line_count '^export BUDGETS=' "$job")" -eq 1 ] ||
        die "job must export one budget: $job"
    [ "$(line_count '^export DISTANCE_MARGIN_COEF=' "$job")" -eq 1 ] ||
        die "job must export one coefficient: $job"
    [ "$(line_count '^source /home/mmoslem3/scratch/attack_if/sbatch/distance_margin_coef_30/_job.sh$' "$job")" -eq 1 ] ||
        die "job does not use the coefficient-ablation runtime: $job"

    job_name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    attack=$(sed -n 's/^export ATTACK=//p' "$job")
    budget=$(sed -n 's/^export BUDGETS=//p' "$job")
    coefficient=$(sed -n 's/^export DISTANCE_MARGIN_COEF=//p' "$job")

    case "$attack" in gradmatch|sapa) ;; *) die "bad attack in $job: $attack" ;; esac
    case "$budget" in 0.001|0.005|0.002) ;; *) die "bad budget in $job: $budget" ;; esac
    case "$coefficient" in 0|0.25|0.5|0.75|1) ;; *) die "bad coefficient in $job: $coefficient" ;; esac

    printf '%s\n' "$job_name" >> "$JOB_NAMES"
    printf '%s\t%s\t%s\n' "$attack" "$budget" "$coefficient" >> "$MATRIX"
done < "$JOB_LIST"

[ "$(sort -u "$JOB_NAMES" | wc -l | tr -d ' ')" -eq "$EXPECTED_JOBS" ] ||
    die "duplicate Slurm job names in $JOB_ROOT"
[ "$(sort -u "$MATRIX" | wc -l | tr -d ' ')" -eq "$EXPECTED_JOBS" ] ||
    die "duplicate attack/budget/coefficient cells in $JOB_ROOT"

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: validated %s jobs (2 attacks x 3 budgets x 5 coefficients)\n' "$EXPECTED_JOBS"
    while IFS= read -r job; do
        printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"
    done < "$JOB_LIST"
    exit 0
fi

mkdir -p "$LOG_ROOT"
ACTIVE_NAMES=$(mktemp)
squeue -h -u "${USER:-mmoslem3}" -o '%.200j' > "$ACTIVE_NAMES"

submitted=0
skipped=0
while IFS= read -r job; do
    job_name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    if grep -Fqx "$job_name" "$ACTIVE_NAMES"; then
        printf 'skip active duplicate: %s\n' "$job_name"
        skipped=$((skipped + 1))
        continue
    fi
    output=$(sbatch --parsable --account="$ACCOUNT" "$job")
    job_id=${output%%;*}
    case "$job_id" in
        *[!0-9]*|'') die "could not parse job ID from sbatch output: $output" ;;
    esac
    printf 'submitted %s -> %s\n' "$job_name" "$job_id"
    printf '%s\n' "$job_name" >> "$ACTIVE_NAMES"
    submitted=$((submitted + 1))
done < "$JOB_LIST"

printf 'Coefficient sweep submission complete: %s submitted, %s active duplicates skipped\n' "$submitted" "$skipped"
