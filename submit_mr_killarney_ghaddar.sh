#!/bin/sh
# Killarney: M/R ablations only; this file submits, but never trains on login.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT="$(cd -- "${SOURCE_ROOT:-$SCRIPT_DIR}" && pwd -P)"
BATCH="$SCRIPT_DIR/sbatch/rebuttal_mr_killarney"
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
[ -d "$DATA_ROOT" ] || die "missing data directory: $DATA_ROOT"
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd -P)"
cd "$BATCH"
sha256sum --check --quiet code.sha256 || die 'prepared code/plan changed'
while IFS=$'\t' read -r relative bytes dataset hash; do
    [ -s "$ROOT/$relative" ] || die "missing surrogate: $ROOT/$relative"
    [ "$(stat -c %s "$ROOT/$relative")" = "$bytes" ] || die "incomplete surrogate: $ROOT/$relative"
done < <(tail -n +2 surrogates.tsv)
while IFS=$'\t' read -r relative bytes dataset hash; do
    [ -s "$DATA_ROOT/$relative" ] || die "missing data: $DATA_ROOT/$relative"
    [ "$(stat -c %s "$DATA_ROOT/$relative")" = "$bytes" ] || die "incomplete data: $DATA_ROOT/$relative"
done < <(tail -n +2 data.tsv)
cluster="$(scontrol show config | awk '$1=="ClusterName"{print $3}')"
[ -n "$cluster" ] || die 'cannot identify the Slurm cluster'
inventory="$(sinfo -h -o '%G' --Node)" || die 'cannot read GPU inventory'
[[ "${inventory,,}" == *l40* ]] || die 'no L40 GPU found in this cluster inventory'
mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another M/R submitter is running'
identity="$cluster aip-bghaddar $ROOT"
if [ -f cluster.txt ]; then
    [ "$(cat cluster.txt)" = "$identity" ] || die 'submission ledger belongs to another cluster/project'
else
    printf '%s\n' "$identity" > cluster.txt
fi
protocol_hash="$(cat plan.json code.sha256 | sha256sum | awk '{print $1}')"
queue="$(squeue -h -u "$USER" -o '%A %T')" || die 'cannot read your queue'
declare -A active ids complete
while read -r number state; do
    [ -z "$number" ] || active["$number"]="$state"
done <<< "$queue"
submitted=0
while IFS=$'\t' read -r task seconds memory requirements; do
    marker="$ROOT/rebuttal_mr_result/jobs/$task/.done"
    if [ -f "$marker" ]; then
        jq -e --arg hash "$protocol_hash" '.protocol_hash==$hash' "$marker" >/dev/null || die "incompatible result: $marker"
        complete["$task"]=1
        printf 'SKIP %s: complete\n' "$task"
        continue
    fi
    dependency=''
    if [ "$requirements" != - ]; then
        IFS=',' read -ra parents <<< "$requirements"
        for parent in "${parents[@]}"; do
            if [ -n "${complete[$parent]:-}" ]; then continue; fi
            [ -n "${ids[$parent]:-}" ] || die "missing upstream job: $parent"
            dependency="${dependency:+$dependency:}${ids[$parent]}"
        done
    fi
    previous='' old_dependency=''
    if [ -f submissions.tsv ]; then
        read -r previous old_dependency < <(awk -F '\t' -v task="$task" '$1==task{job=$2;dep=$3} END{print job, dep}' submissions.tsv)
    fi
    if [ -n "$previous" ] && [ -n "${active[$previous]:-}" ]; then
        if [ "${active[$previous]}" = PENDING ] && [ -n "$dependency" ] && [ "$old_dependency" != "$dependency" ]; then
            scontrol update JobId="$previous" Dependency="afterok:$dependency"
            printf '%s\t%s\t%s\t%s\n' "$task" "$previous" "$dependency" "$(date -u +%FT%TZ)" >> submissions.tsv
            printf 'UPDATED %s: dependencies now %s\n' "$task" "$dependency"
        fi
        ids["$task"]="$previous"
        printf 'SKIP %s: job %s already queued/running\n' "$task" "$previous"
        continue
    fi
    arguments=(--parsable --account=aip-bghaddar --nodes=1 --ntasks=1
        --cpus-per-task=1 --mem="$memory" --gres=gpu:1
        --time="$(printf '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))")"
        --signal=USR1@180 --job-name="mrk_$task" --chdir="$ROOT"
        --output="$BATCH/logs/%x-%j.out")
    [ -z "$dependency" ] || arguments+=(--dependency="afterok:$dependency")
    number="$(sbatch "${arguments[@]}" "$BATCH/job.sh" "$ROOT" "$BATCH" "$task" "$ENV_ACTIVATE" "$DATA_ROOT")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID: $number"
    ids["$task"]="$number"
    printf '%s\t%s\t%s\t%s\n' "$task" "$number" "${dependency:--}" "$(date -u +%FT%TZ)" >> submissions.tsv
    printf 'SUBMITTED %s: %s, aip-bghaddar, %ss, %s, gpu:1\n' "$task" "$number" "$seconds" "$memory"
    submitted=$((submitted+1))
done < <(jq -r '.tasks[] | [.id,(.seconds|tostring),.memory,(if (.requires|length)>0 then (.requires|join(",")) else "-" end)] | @tsv' plan.json)
printf 'Submitted %s jobs. Full plan: 1 preflight + 28 experiment jobs for 12 cells.\n' "$submitted"
printf 'Rerun this script to resume unfinished work. Combined results: %s/rebuttal_mr_result/summaries/\n' "$ROOT"
