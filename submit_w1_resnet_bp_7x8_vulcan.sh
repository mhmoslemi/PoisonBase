#!/bin/sh
# Submit only the ten new ResNet20 BP cells in tab:reb-access.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_access_res_bp_7x8_vulcan"
RESULTS="$ROOT/rebuttal_access_res_bp_7x8_result"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for program in sbatch squeue sinfo jq flock rsync sha256sum; do
    command -v "$program" >/dev/null || die "missing command: $program"
done
[ -s "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
for file in data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch batches.meta; do
    [ -s "$ROOT/data/cifar-10-batches-py/$file" ] || die "missing data: $file"
done
cd "$BATCH"
sha256sum --check --quiet code.sha256 || die 'prepared code/plan changed'
cd "$ROOT"
sha256sum --check --quiet "$BATCH/inputs.sha256" || die 'a required saved surrogate or target file is missing or changed'
got="$(jq -c '.pairs["dog-bird"].indices[:7]' target_sets/ResNet20BN_fc_dog-bird.json)"
[ "$got" = '[3495,2903,6697,2315,6899,2312,6353]' ] || die 'ResNet20 BP pinned targets changed'
inventory="$(sinfo -h -o '%G' --Node)" || die 'cannot read GPU inventory'
[[ "$inventory" == *gpu:l40s:* ]] || die 'Vulcan L40S GPUs were not found'
mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another ResNet20 BP submitter is running'
queue="$(squeue -h -u "$USER" -o '%A')" || die 'cannot read your queue'
declare -A active
while IFS= read -r id; do [ -z "$id" ] || active["$id"]=1; done <<< "$queue"
submitted=0
while IFS=$'\t' read -r task attack method condition start stop; do
    [ "$task" != id ] || continue
    if [ -f "$RESULTS/jobs/$task/.done" ]; then
        printf 'SKIP task %s: complete\n' "$task"
        continue
    fi
    previous=''
    if [ -f "$BATCH/submissions.tsv" ]; then
        previous="$(awk -F '\t' -v task="$task" '$1==task{job=$2} END{print job}' "$BATCH/submissions.tsv")"
    fi
    if [ -n "$previous" ] && [ -n "${active[$previous]:-}" ]; then
        printf 'SKIP task %s: job %s already queued/running\n' "$task" "$previous"
        continue
    fi
    number="$(sbatch --parsable --account=aip-bghaddar \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time=04:00:00 --job-name="w1r_bp_${method}_${condition}_${start}" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        "$BATCH/job.sh" "$ROOT" "$BATCH" "$task" "$ENV_ACTIVATE")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID: $number"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$number" "$method" "$condition" "$start" "$stop" >> "$BATCH/submissions.tsv"
    printf 'SUBMITTED task %s: %s, %s, condition %s, draws %s:%s\n' "$task" "$number" "$method" "$condition" "$start" "$stop"
    submitted=$((submitted+1))
done < "$BATCH/manifest.tsv"
printf 'Submitted %s jobs. Full plan: 41 jobs, 4h each, aip-bghaddar, 1 CPU, 6G, 1 L40S.\n' "$submitted"
printf 'Rerun this script to resume unfinished tasks. Results: %s\n' "$RESULTS"
