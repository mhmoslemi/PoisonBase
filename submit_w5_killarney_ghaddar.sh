#!/bin/sh
# Run with sh from the Killarney checkout, normally ~/scratch/attack_if.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT="$(cd -- "${SOURCE_ROOT:-$SCRIPT_DIR}" && pwd -P)"
BATCH="$SCRIPT_DIR/sbatch/rebuttal_w5_killarney"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for program in sbatch squeue scontrol sinfo jq flock rsync sha256sum stat; do
    command -v "$program" >/dev/null || die "missing command: $program"
done
[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
if [ -z "${DATA_ROOT:-}" ]; then
    DATA_ROOT="$ROOT/data"
    [ -f "$DATA_ROOT/cifar-10-batches-py/data_batch_1" ] || DATA_ROOT=/home/mmoslem3/scratch/data
fi
for file in data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch batches.meta; do
    [ -s "$DATA_ROOT/cifar-10-batches-py/$file" ] || die "missing CIFAR-10 file: $DATA_ROOT/cifar-10-batches-py/$file"
done
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd -P)"
cd "$BATCH"
sha256sum --check --quiet code.sha256 || die 'W5 code/plan differs from the prepared snapshot'
missing=0
while IFS=$'\t' read -r relative bytes kind; do
    if [ ! -s "$ROOT/$relative" ] || [ "$(stat -c %s "$ROOT/$relative")" != "$bytes" ]; then
        if [ "$missing" -lt 10 ]; then printf 'MISSING/WRONG SIZE: %s\n' "$ROOT/$relative" >&2; fi
        missing=$((missing+1))
    fi
done < <(tail -n +2 "$BATCH/artifacts.tsv")
[ "$missing" -eq 0 ] || die "$missing input files missing or incomplete; use the two vulcan_*_files.txt transfer lists"

cluster="$(scontrol show config | awk '$1=="ClusterName"{print $3; exit}')"
[ -n "$cluster" ] || die 'cannot identify the Slurm cluster'
inventory="$(sinfo -h -o '%G' --Node)" || die 'cannot read GPU inventory'
[ -n "$inventory" ] || die 'no GPU inventory returned by Slurm'
mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another W5 submitter is running'
identity="$cluster aip-bghaddar $ROOT"
if [ -f "$BATCH/cluster.txt" ]; then
    [ "$(cat "$BATCH/cluster.txt")" = "$identity" ] || die 'submission ledger belongs to a different cluster/project'
else
    printf '%s\n' "$identity" > "$BATCH/cluster.txt"
fi
queue="$(squeue -h -u "$USER" -o '%A')" || die 'cannot read the Slurm queue'
declare -A active
while IFS= read -r number; do
    [ -z "$number" ] || active["$number"]=1
done <<< "$queue"
submitted=0
while IFS= read -r task; do
    if [ -f "$ROOT/rebuttal_w5_result/jobs/$task/.done" ]; then
        printf 'SKIP %s: complete\n' "$task"
        continue
    fi
    previous=''
    if [ -f "$BATCH/submissions.tsv" ]; then
        previous="$(awk -F '\t' -v task="$task" '$1==task{job=$2} END{print job}' "$BATCH/submissions.tsv")"
    fi
    if [ -n "$previous" ] && [ -n "${active[$previous]:-}" ]; then
        printf 'SKIP %s: job %s already queued/running\n' "$task" "$previous"
        continue
    fi
    number="$(sbatch --parsable --account=aip-bghaddar \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:1 \
        --time=03:00:00 --signal=USR1@180 --job-name="w5k_$task" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        "$BATCH/job.sh" "$ROOT" "$BATCH" "$task" "$ENV_ACTIVATE" "$DATA_ROOT")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID: $number"
    printf '%s\t%s\t%s\n' "$task" "$number" "$(date -u +%FT%TZ)" >> "$BATCH/submissions.tsv"
    printf 'SUBMITTED %s: %s, aip-bghaddar, 3h, 1 CPU, 6G, gpu:1\n' "$task" "$number"
    submitted=$((submitted+1))
done < <(jq -r '.tasks[].id' "$BATCH/plan.json")
printf 'Submitted %s jobs; full W5 plan: %s jobs, 1,446 final victims.\n' "$submitted" "$(jq '.tasks|length' "$BATCH/plan.json")"
printf 'Rerun this same script to resume unfinished jobs. Results: %s/rebuttal_w5_result\n' "$ROOT"
