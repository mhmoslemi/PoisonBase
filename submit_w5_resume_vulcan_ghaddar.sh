#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_w5_killarney"
RESUME="$ROOT/sbatch/rebuttal_w5_resume_vulcan"
cd "$BATCH"
sha256sum --check --quiet code.sha256
[ -f /home/mmoslem3/ENV/bin/activate ]
for file in data_batch_{1..5} test_batch batches.meta; do [ -s "$ROOT/data/cifar-10-batches-py/$file" ]; done
cluster="$(scontrol show config | awk '$1=="ClusterName"{print $3}')"
[[ "${cluster,,}" == *vulcan* ]]
inventory="$(sinfo -h -o '%G' --Node)"
[[ "$inventory" == *gpu:l40s:* ]]
[ "$(wc -l < "$RESUME/tasks.tsv")" -eq 22 ]
mkdir -p "$RESUME/logs"
exec 9>"$RESUME/submission.lock"
flock -n 9 || { echo 'Another W5 resume submitter is running.' >&2; exit 1; }
queue="$(squeue -h -u "$USER" -o '%A %j')"
declare -A active names
while read -r id name; do
    if [ -n "$id" ]; then active["$id"]=1; [ -z "$name" ] || names["$name"]="$id"; fi
done <<< "$queue"
submitted=0
while IFS=$'\t' read -r task seconds remaining; do
    [[ "$seconds" == 10800 || "$seconds" == 14400 ]]
    jq -e --arg task "$task" '.tasks[] | select(.id==$task and .model=="ResNet20BN")' plan.json >/dev/null
    if [ -f "$ROOT/rebuttal_w5_result/jobs/$task/.done" ]; then echo "SKIP $task: complete"; continue; fi
    previous=''
    if [ -f "$RESUME/submissions.tsv" ]; then previous="$(awk -F '\t' -v task="$task" '$1==task{job=$2} END{print job}' "$RESUME/submissions.tsv")"; fi
    if [ -n "$previous" ] && [ -n "${active[$previous]:-}" ]; then echo "SKIP $task: $previous active"; continue; fi
    if [ -n "${names[w5v_$task]:-}" ] || [ -n "${names[w5k_$task]:-}" ]; then echo "SKIP $task: already queued/running"; continue; fi
    hours=$(( seconds / 3600 ))
    id="$(sbatch --parsable --account=aip-bghaddar --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="0${hours}:00:00" \
        --signal=USR1@180 --job-name="w5v_$task" --chdir="$ROOT" \
        --output="$RESUME/logs/%x-%j.out" "$RESUME/job.sh" "$ROOT" "$task" "$seconds")"
    id="${id%%;*}"
    [[ "$id" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\n' "$task" "$id" "$(date -u +%FT%TZ)" >> "$RESUME/submissions.tsv"
    echo "SUBMITTED $task: $id, aip-bghaddar, ${hours}h, 6G, $remaining unfinished victims"
    submitted=$((submitted+1))
done < "$RESUME/tasks.tsv"
echo "Submitted $submitted W5 resume jobs."
