#!/bin/sh
# Six independent Meta-Grad K=2 ablation cells, 8 targets x 5 victims each.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT="$(cd -- "${SOURCE_ROOT:-$SCRIPT_DIR}" && pwd -P)"
BATCH="$SCRIPT_DIR/sbatch/rebuttal_meta_k2_vulcan"
RESULTS="$ROOT/rebuttal_meta_k2_result"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for program in sbatch squeue scontrol sinfo jq flock sha256sum stat rsync; do
    command -v "$program" >/dev/null || die "missing command: $program"
done
[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
cd "$BATCH"
sha256sum --check --quiet code.sha256 || die 'the frozen code or plan changed'
while IFS=$'\t' read -r relative bytes; do
    [ -s "$ROOT/$relative" ] || die "missing input: $ROOT/$relative"
    [ "$(stat -Lc %s "$ROOT/$relative")" = "$bytes" ] || die "incomplete input: $ROOT/$relative"
done < <(tail -n +2 inputs.tsv)
cluster="$(scontrol show config | awk '$1=="ClusterName"{print $3}')"
[[ "${cluster,,}" == *vulcan* ]] || die 'run this submitter on Vulcan'
inventory="$(sinfo -h -o '%G' --Node)"
[[ "$inventory" == *gpu:l40s:* ]] || die 'L40S GPU request is unavailable'
mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another K=2 submitter is running'
identity="$cluster aip-yiweilu $ROOT"
if [ -f cluster.txt ]; then
    [ "$(cat cluster.txt)" = "$identity" ] || die 'submission ledger belongs to another cluster/project'
else
    printf '%s\n' "$identity" > cluster.txt
fi
queue="$(squeue -h -u "$USER" -o '%A %T')"
declare -A live
while read -r number state; do
    [ -z "$number" ] || live["$number"]="$state"
done <<< "$queue"
submitted=0
while IFS= read -r record; do
    task="$(jq -r '.id' <<< "$record")"
    if [ -f "$RESULTS/jobs/$task/.done" ]; then
        jq -e --argjson task "$record" '.task==$task' "$RESULTS/jobs/$task/task.json" >/dev/null || die "incompatible completed task: $task"
        printf 'SKIP %s: complete\n' "$task"
        continue
    fi
    previous=''
    if [ -f submissions.tsv ]; then
        previous="$(awk -F '\t' -v task="$task" '$1==task{job=$2} END{print job}' submissions.tsv)"
    fi
    if [ -n "$previous" ] && [ -n "${live[$previous]:-}" ]; then
        printf 'SKIP %s: job %s already queued/running\n' "$task" "$previous"
        continue
    fi
    number="$(sbatch --parsable --account=aip-yiweilu --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time=03:00:00 \
        --signal=B:USR1@180 --job-name="mk2_$task" --chdir="$ROOT" \
        --output="$BATCH/logs/%x-%j.out" "$BATCH/job.sh" "$ROOT" "$task" "$ENV_ACTIVATE")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID: $number"
    printf '%s\t%s\t%s\n' "$task" "$number" "$(date -u +%FT%TZ)" >> submissions.tsv
    printf 'SUBMITTED %s: %s, aip-yiweilu, 3h, 1 CPU, 6G, 1 L40S\n' "$task" "$number"
    submitted=$((submitted+1))
done < <(jq -c '.tasks[]' plan.json)
printf 'Submitted %s jobs (six cells total). Results: %s\n' "$submitted" "$RESULTS"
