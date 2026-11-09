#!/bin/bash
# Submit only Reviewer 2 W4; running this file is the submission action.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_w4"
PLAN="$BATCH/plan.json"
RESULTS="$ROOT/rebuttal_w4_result"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
ACCOUNT="${W4_ACCOUNT:-aip-bghaddar}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for program in sbatch squeue jq flock rsync sha256sum; do
    command -v "$program" >/dev/null || die "missing command: $program"
done
[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
[ -d "$ROOT/data/cifar-10-batches-py" ] || die 'missing CIFAR-10 data'
cd "$BATCH"
sha256sum --check --quiet code.sha256 || die 'frozen W4 inputs changed'
[ "$(jq '.tasks|length' "$PLAN")" -eq 44 ] || die 'expected 44 W4 tasks'
tasks="$(jq -r '.tasks[].id' "$PLAN")"
if [ "$#" -gt 0 ]; then
    tasks="$(printf '%s\n' "$@")"
    for task in "$@"; do
        jq -e --arg task "$task" 'any(.tasks[]; .id==$task)' "$PLAN" >/dev/null || die "unknown W4 task: $task"
    done
fi
for model in ConvNetBN ResNet20BN; do
    got="$(jq -c '.pairs["dog-bird"].indices' "$ROOT/target_sets/${model}_gradmatch_dog-bird.json")"
    expected="$(jq -c --arg model "$model" '.targets[$model]' "$PLAN")"
    [ "$got" = "$expected" ] || die "pinned targets changed for $model"
    for ((index=0; index<5; index++)); do
        path="$ROOT/cache/surrogates/${model}_60ep_lr0.1_bs128_seed42/net_$index.pt"
        [ -s "$path" ] || die "missing trained crafting surrogate: $path"
    done
done
mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another W4 submitter is running'
declare -A active
queue="$(squeue -h -u "$USER" -o '%A')" || die 'cannot read the Slurm queue'
while IFS= read -r number; do
    [ -z "$number" ] || active["$number"]=1
done <<< "$queue"
submitted=0
while IFS= read -r task; do
    if [ -f "$RESULTS/jobs/$task/.done" ]; then
        printf 'SKIP %s: complete\n' "$task"
        continue
    fi
    if [ -f "$ROOT/sbatch/rebuttal_w4_90m/plan.json" ] && \
        jq -e --arg task "$task" 'any(.tasks[]; .parent_task==$task)' "$ROOT/sbatch/rebuttal_w4_90m/plan.json" >/dev/null; then
        W4_ACCOUNT="$ACCOUNT" sh "$ROOT/submit_w4_90m_ghaddar.sh" "$task"
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
        --time=05:00:00 --signal=B:USR1@180 --job-name="w4_$task" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        "$BATCH/job.sh" "$ROOT" "$task" "$ENV_ACTIVATE")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID: $number"
    printf '%s\t%s\t%s\n' "$task" "$number" "$(date -u +%FT%TZ)" >> "$BATCH/submissions.tsv"
    printf 'SUBMITTED %s: job=%s account=%s time=5h CPU=1 RAM=6G GPU=L40S:1\n' "$task" "$number" "$ACCOUNT"
    submitted=$((submitted+1))
done <<< "$tasks"
printf 'Submitted %s jobs (full W4 batch: 12 First-m + 32 Rand = 44).\n' "$submitted"
printf 'Rerun this script to resume unfinished tasks. Results: %s\n' "$RESULTS"
