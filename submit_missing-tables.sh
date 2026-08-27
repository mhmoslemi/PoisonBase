#!/usr/bin/env bash
# Submit exactly the still-blank cells in full.tex, defense.tex, and
# cross-table.tex. defense-extra.tex is complete and therefore has no job in
# this set. Jobs are ordered by requested walltime, shortest first. DRY_RUN=1
# validates and prints every sbatch command without submitting it.

# Keep the common `sh submit_missing-tables.sh` invocation working even on
# systems where /bin/sh is dash rather than Bash.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/mmoslem3/scratch/PoisonBase}"
ACCOUNT="${ACCOUNT:-aip-boyuwang}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
JOB_ROOT="$SOURCE_ROOT/sbatch/missing_tables_20260827"
LOG_ROOT="$SOURCE_ROOT/sbatch/logs2"
EXPECTED_FULL=8
EXPECTED_DEFENSE=1
EXPECTED_CROSS=23
EXPECTED_TOTAL=32

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

job_count() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name '*.sh' | wc -l | tr -d ' '
}

kind_count() {
    pattern="$1"
    find "$JOB_ROOT" -maxdepth 1 -type f -name "$pattern" | wc -l | tr -d ' '
}

line_count() {
    pattern="$1"
    file="$2"
    count=$(grep -c "$pattern" "$file" 2>/dev/null || true)
    printf '%s\n' "$count"
}

requested_seconds() {
    requested="$1"
    printf '%s\n' "$requested" | awk -F '[-:]' \
        '{ print (($1 * 24 + $2) * 60 + $3) * 60 + $4 }'
}

ordered_jobs() {
    find "$JOB_ROOT" -maxdepth 1 -type f -name '*.sh' -print |
    while IFS= read -r job; do
        requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
        [ -n "$requested" ] || die "missing #SBATCH --time in $job"
        seconds=$(requested_seconds "$requested")
        printf '%010d\t%s\n' "$seconds" "$job"
    done | sort -n -k1,1 -k2,2 | cut -f2-
}

validate_jobs() {
    [ -d "$JOB_ROOT" ] || die "job directory missing: $JOB_ROOT"
    [ "$(job_count)" -eq "$EXPECTED_TOTAL" ] || \
        die "expected $EXPECTED_TOTAL one-cell jobs; found $(job_count)"
    [ "$(kind_count 'b2040_attack_*.sh')" -eq "$EXPECTED_FULL" ] || \
        die "expected $EXPECTED_FULL full.tex jobs"
    [ "$(kind_count 'missing_defense_*.sh')" -eq "$EXPECTED_DEFENSE" ] || \
        die "expected $EXPECTED_DEFENSE defense.tex job"
    [ "$(kind_count 'cross_*.sh')" -eq "$EXPECTED_CROSS" ] || \
        die "expected $EXPECTED_CROSS cross-table.tex jobs"

    job_names=$(mktemp)
    trap 'unlink "$job_names" 2>/dev/null || true' EXIT HUP INT TERM

    find "$JOB_ROOT" -maxdepth 1 -type f -name '*.sh' -print |
    while IFS= read -r job; do
        base=$(basename "$job")
        [ "$(line_count '^#SBATCH --job-name=' "$job")" -eq 1 ] || \
            die "expected one job name in $job"
        [ "$(line_count '^#SBATCH --time=' "$job")" -eq 1 ] || \
            die "expected one requested time in $job"
        [ "$(line_count '^#SBATCH --cpus-per-task=1$' "$job")" -eq 1 ] || \
            die "job does not request exactly one CPU: $job"
        [ "$(line_count '^#SBATCH --mem=7G$' "$job")" -eq 1 ] || \
            die "job does not request 7 GB: $job"
        [ "$(line_count '^#SBATCH --gpus-per-node=l40s:1$' "$job")" -eq 1 ] || \
            die "job does not request one Vulcan L40S: $job"
        [ "$(line_count '^#SBATCH --mail' "$job")" -eq 0 ] || \
            die "email directive found in $job"
        [ "$(line_count '^#SBATCH --output=/home/mmoslem3/scratch/PoisonBase/sbatch/logs2/' "$job")" -eq 1 ] || \
            die "job output is not routed to sbatch/logs2: $job"
        [ "$(line_count '^export SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase$' "$job")" -eq 1 ] || \
            die "incorrect source root in $job"
        sed -n 's/^#SBATCH --job-name=//p' "$job" >> "$job_names"

        requested=$(sed -n 's/^#SBATCH --time=//p' "$job")
        seconds=$(requested_seconds "$requested")
        if [ "$seconds" -gt 25200 ]; then
            case "$base" in
                b2040_attack_*.sh) ;;
                *) die "non-attack job exceeds the seven-hour ceiling: $job" ;;
            esac
        fi

        case "$base" in
            b2040_attack_*.sh)
                [ "$(line_count '^export JOB_KIND=attack$' "$job")" -eq 1 ] || \
                    die "full.tex job is not a single attack: $job"
                [ "$(line_count '^export BUDGETS=[^[:space:]]\+$' "$job")" -eq 1 ] || \
                    die "full.tex job does not specify exactly one budget: $job"
                [ "$(line_count '^export NUM_TARGETS=8$' "$job")" -eq 1 ] || \
                    die "full.tex cell does not use eight targets: $job"
                [ "$(line_count '^export NUM_VICTIMS=6$' "$job")" -eq 1 ] || \
                    die "full.tex cell does not use six victims: $job"
                [ "$(line_count '^source /home/mmoslem3/scratch/PoisonBase/sbatch/_job_common.sh$' "$job")" -eq 1 ] || \
                    die "full.tex job uses the wrong runtime: $job"
                ;;
            missing_defense_*.sh)
                [ "$(line_count '^export JOB_KIND=defense$' "$job")" -eq 1 ] || \
                    die "defense.tex job is not a single defense: $job"
                [ "$(line_count '^export BUDGETS=[^[:space:]]\+$' "$job")" -eq 1 ] || \
                    die "defense.tex job does not specify exactly one budget: $job"
                [ "$(line_count '^export SELS=[^[:space:]]\+$' "$job")" -eq 1 ] || \
                    die "defense.tex job does not specify exactly one selection: $job"
                [ "$(line_count '^export DEFENSES=[^[:space:]]\+$' "$job")" -eq 1 ] || \
                    die "defense.tex job does not specify exactly one defense: $job"
                [ "$(line_count '^export NUM_TARGETS=7$' "$job")" -eq 1 ] || \
                    die "defense.tex cell does not use seven targets: $job"
                [ "$(line_count '^export NUM_VICTIMS=5$' "$job")" -eq 1 ] || \
                    die "defense.tex cell does not use five victims: $job"
                [ "$(line_count '^source /home/mmoslem3/scratch/PoisonBase/sbatch/_job_common.sh$' "$job")" -eq 1 ] || \
                    die "defense.tex job uses the wrong runtime: $job"
                ;;
            cross_*.sh)
                [ "$(line_count '^export CROSS_RUN_NAME=' "$job")" -eq 1 ] || \
                    die "cross job does not identify exactly one result cell: $job"
                [ "$(line_count '^export CROSS_NUM_TARGETS=5$' "$job")" -eq 1 ] || \
                    die "cross cell does not use five targets: $job"
                [ "$(line_count '^export CROSS_NUM_VICTIMS=4$' "$job")" -eq 1 ] || \
                    die "cross cell does not use four victims: $job"
                [ "$(line_count '^source /home/mmoslem3/scratch/PoisonBase/sbatch/_cross_job_common.sh$' "$job")" -eq 1 ] || \
                    die "cross job uses the wrong runtime: $job"
                ;;
            *) die "unrecognized job file: $job" ;;
        esac
    done

    [ "$(sort -u "$job_names" | wc -l | tr -d ' ')" -eq "$EXPECTED_TOTAL" ] || \
        die "duplicate Slurm job names in $JOB_ROOT"
    unlink "$job_names"
    trap - EXIT HUP INT TERM
}

validate_jobs

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY RUN: %s one-cell jobs validated; submission order is shortest first\n' \
        "$EXPECTED_TOTAL"
    ordered_jobs | while IFS= read -r job; do
        printf 'sbatch --account=%s %s\n' "$ACCOUNT" "$job"
    done
    exit 0
fi

case "$(hostname -s)" in
    vulcan*) ;;
    *) die "run this submitter on Vulcan (current host: $(hostname -s))" ;;
esac

[ -f "$ENV_ACTIVATE" ] || die "Python environment missing: $ENV_ACTIVATE"
[ -d "$SOURCE_ROOT/data/cifar-10-batches-py" ] || \
    die "CIFAR-10 input missing under $SOURCE_ROOT/data"
[ -f "$SOURCE_ROOT/sbatch/_job_common.sh" ] || \
    die "missing runtime: $SOURCE_ROOT/sbatch/_job_common.sh"
[ -f "$SOURCE_ROOT/sbatch/_cross_job_common.sh" ] || \
    die "missing runtime: $SOURCE_ROOT/sbatch/_cross_job_common.sh"
mkdir -p "$LOG_ROOT"

ordered=$(mktemp)
active_names=$(mktemp)
trap 'unlink "$ordered" 2>/dev/null || true; unlink "$active_names" 2>/dev/null || true' EXIT HUP INT TERM
ordered_jobs > "$ordered"
squeue -h -u "${USER:-mmoslem3}" -o '%.200j' > "$active_names"

submitted=0
skipped=0
while IFS= read -r job; do
    job_name=$(sed -n 's/^#SBATCH --job-name=//p' "$job")
    if grep -Fqx "$job_name" "$active_names"; then
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
    printf '%s\n' "$job_name" >> "$active_names"
    submitted=$((submitted + 1))
done < "$ordered"

printf 'Missing-table submission complete: %s submitted, %s active duplicates skipped\n' \
    "$submitted" "$skipped"
