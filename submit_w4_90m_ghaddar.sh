#!/bin/bash
# Resume only the unfinished W4 targets split into 90-minute jobs.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_w4_90m"
PLAN="$BATCH/plan.json"
ACCOUNT="${W4_ACCOUNT:-aip-bghaddar}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
cd "$BATCH"
sha256sum --check --quiet inputs.sha256 || die '90-minute inputs changed'
(cd "$ROOT/sbatch/rebuttal_w4" && sha256sum --check --quiet code.sha256) || die 'original W4 inputs changed'
[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
tasks="$(jq -r '.tasks[].id' "$PLAN")"
if [ "$#" -gt 0 ]; then
    tasks=''
    for requested in "$@"; do
        selected="$(jq -r --arg requested "$requested" '.tasks[] | select(.id==$requested or .parent_task==$requested) | .id' "$PLAN")"
        [ -n "$selected" ] || die "unknown split task or parent: $requested"
        tasks="$(printf '%s\n%s\n' "$tasks" "$selected")"
    done
    tasks="$(printf '%s\n' "$tasks" | awk 'NF && !seen[$0]++')"
fi
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another 90-minute submitter is running'
declare -A active
queue="$(squeue -h -u "$USER" -o '%A')" || die 'cannot read the Slurm queue'
while IFS= read -r number; do
    [ -z "$number" ] || active["$number"]=1
done <<< "$queue"
while IFS= read -r number; do
    [ -z "${active[$number]:-}" ] || die "original job $number is still active; do not overlap the replacement jobs"
done < <(jq -r '.parents[].old_job' "$BATCH/split_metadata.json" | sort -u)
submitted=0
while IFS= read -r task; do
    if [ -f "$ROOT/rebuttal_w4_result/jobs/$task/.done" ]; then
        printf 'SKIP %s: complete\n' "$task"
        continue
    fi
    last_job=''
    if [ -f "$BATCH/submissions.tsv" ]; then
        last_job="$(awk -F '\t' -v task="$task" '$1==task{job=$2} END{print job}' "$BATCH/submissions.tsv")"
    fi
    if [ -n "$last_job" ] && [ -n "${active[$last_job]:-}" ]; then
        printf 'SKIP %s: job %s already queued/running\n' "$task" "$last_job"
        continue
    fi
    number="$(sbatch --parsable --account="$ACCOUNT" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time=01:30:00 --signal=B:USR1@180 --job-name="w4_90_$task" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        "$BATCH/job.sh" "$ROOT" "$task" "$ENV_ACTIVATE")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID: $number"
    printf '%s\t%s\t%s\n' "$task" "$number" "$(date -u +%FT%TZ)" >> "$BATCH/submissions.tsv"
    printf 'SUBMITTED %s: job=%s account=%s time=90m CPU=1 RAM=6G GPU=L40S:1\n' "$task" "$number" "$ACCOUNT"
    submitted=$((submitted+1))
done <<< "$tasks"
printf 'Submitted %s 90-minute W4 jobs. Completed targets and victims are reused.\n' "$submitted"
