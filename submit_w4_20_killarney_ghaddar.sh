#!/bin/sh
# Only the 20 W4 tasks formerly submitted as 2175914..2175934, excluding 2175917.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_w4"
PLAN="$BATCH/plan.json"
RESULTS="$ROOT/rebuttal_w4_result"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for program in sbatch squeue scontrol sinfo jq flock rsync sha256sum; do
    command -v "$program" >/dev/null || die "missing command: $program"
done
[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
if [ -z "${DATA_ROOT:-}" ]; then
    DATA_ROOT="$ROOT/data"
    [ -d "$DATA_ROOT/cifar-10-batches-py" ] || DATA_ROOT=/home/mmoslem3/scratch/data
fi
for file in data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch batches.meta; do
    [ -s "$DATA_ROOT/cifar-10-batches-py/$file" ] || die "missing data: $DATA_ROOT/cifar-10-batches-py/$file"
done
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd -P)"
cd "$BATCH"
sha256sum --check --quiet code.sha256 || die 'frozen W4 inputs changed'
tasks=(rand_conv_sapa_b001_2 first_conv_sapa_b002 rand_conv_sapa_b002_1
    first_conv_sapa_b005 rand_conv_sapa_b005_1 rand_conv_sapa_b005_2
    first_res_gm_b001 rand_res_gm_b001_1 rand_res_gm_b001_2 rand_res_gm_b001_3
    first_res_gm_b002 rand_res_gm_b002_1 rand_res_gm_b002_2 rand_res_gm_b002_3
    first_res_gm_b005 rand_res_gm_b005_1 rand_res_gm_b005_2 rand_res_gm_b005_3
    rand_res_gm_b005_4 first_res_sapa_b001)
[ "${#tasks[@]}" -eq 20 ] || die 'expected exactly 20 tasks'
for task in "${tasks[@]}"; do
    jq -e --arg task "$task" 'any(.tasks[]; .id==$task)' "$PLAN" >/dev/null || die "unknown task: $task"
done
for model in ConvNetBN ResNet20BN; do
    got="$(jq -c '.pairs["dog-bird"].indices' "$ROOT/target_sets/${model}_gradmatch_dog-bird.json")"
    expected="$(jq -c --arg model "$model" '.targets[$model]' "$PLAN")"
    [ "$got" = "$expected" ] || die "pinned targets changed: $model"
    for ((index=0; index<5; index++)); do
        file="$ROOT/cache/surrogates/${model}_60ep_lr0.1_bs128_seed42/net_$index.pt"
        [ -s "$file" ] || die "missing crafting surrogate: $file"
    done
done
cluster="$(scontrol show config | awk '$1=="ClusterName"{print $3}')"
[ -n "$cluster" ] || die 'cannot identify cluster'
inventory="$(sinfo -h -o '%G' --Node)" || die 'cannot read GPU inventory'
[[ "${inventory,,}" == *l40* ]] || die 'no L40 GPU found in cluster inventory'
mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another W4 submitter is running'
identity="$cluster aip-bghaddar $ROOT"
if [ -f killarney_20_cluster.txt ]; then
    [ "$(cat killarney_20_cluster.txt)" = "$identity" ] || die 'Killarney ledger belongs to another cluster/project'
else
    printf '%s\n' "$identity" > killarney_20_cluster.txt
fi
declare -A active
queue="$(squeue -h -u "$USER" -o '%A')" || die 'cannot read your queue'
while IFS= read -r number; do
    [ -z "$number" ] || active["$number"]=1
done <<< "$queue"
submitted=0
for task in "${tasks[@]}"; do
    if [ -f "$RESULTS/jobs/$task/.done" ]; then
        printf 'SKIP %s: complete\n' "$task"
        continue
    fi
    previous=''
    if [ -f submissions_killarney_20.tsv ]; then
        previous="$(awk -F '\t' -v task="$task" '$1==task{job=$2} END{print job}' submissions_killarney_20.tsv)"
    fi
    if [ -n "$previous" ] && [ -n "${active[$previous]:-}" ]; then
        printf 'SKIP %s: job %s already queued/running\n' "$task" "$previous"
        continue
    fi
    number="$(sbatch --parsable --account=aip-bghaddar \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:1 \
        --time=05:00:00 --signal=B:USR1@180 --job-name="w4k_$task" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        "$BATCH/killarney_job.sh" "$ROOT" "$task" "$ENV_ACTIVATE" "$DATA_ROOT")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID: $number"
    printf '%s\t%s\t%s\n' "$task" "$number" "$(date -u +%FT%TZ)" >> submissions_killarney_20.tsv
    printf 'SUBMITTED %s: %s, aip-bghaddar, 5h, 1 CPU, 6G, gpu:1\n' "$task" "$number"
    submitted=$((submitted+1))
done
printf 'Submitted %s jobs from the selected 20. Rerun this script to resume unfinished tasks.\n' "$submitted"
