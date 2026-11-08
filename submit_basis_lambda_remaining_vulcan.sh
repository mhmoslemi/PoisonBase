#!/bin/sh
# Vulcan replacement: 20 Ghaddar + 13 Wang jobs, maximum six hours each.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT="$(cd -- "${SOURCE_ROOT:-$SCRIPT_DIR}" && pwd -P)"
BATCH="$SCRIPT_DIR/sbatch/rebuttal_basis_lambda_extra_vulcan"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
SITE=vulcan
GPU_REQUEST=gpu:l40s:1
LEDGER="submissions_$SITE.tsv"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for program in sbatch squeue scontrol sinfo jq flock rsync sha256sum stat module; do
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
    source_relative="$relative"
    [ -s "$ROOT/$source_relative" ] || die "missing surrogate: $ROOT/$source_relative"
    [ "$(stat -c %s "$ROOT/$source_relative")" = "$bytes" ] || die "incomplete surrogate: $ROOT/$source_relative"
done < <(tail -n +2 surrogates.tsv)
while IFS=$'\t' read -r relative bytes dataset hash; do
    [ -s "$DATA_ROOT/$relative" ] || die "missing data: $DATA_ROOT/$relative"
    [ "$(stat -c %s "$DATA_ROOT/$relative")" = "$bytes" ] || die "incomplete data: $DATA_ROOT/$relative"
done < <(tail -n +2 data.tsv)
cluster="$(scontrol show config | awk '$1=="ClusterName"{print $3}')"
[ -n "$cluster" ] || die 'cannot identify the Slurm cluster'
[[ "${cluster,,}" == *"$SITE"* ]] || die "run the $SITE script on $SITE (current cluster: $cluster)"
inventory="$(sinfo -h -o '%G' --Node)" || die 'cannot read GPU inventory'
[[ "${inventory,,}" == *gpu:l40s:* ]] || die 'no L40S GPU advertised by Vulcan'
for spec in StdEnv/2023 gcc/12.3 python/3.11.5 cuda/12.6 cudnn/9.10.0.56; do
    module spider "$spec" >/dev/null 2>&1 || die "required module unavailable: $spec"
done
mkdir -p "$BATCH/logs/$SITE"
exec 9>"$BATCH/submission_$SITE.lock"
flock -n 9 || die 'another BASIS lambda submitter is running'
identity="$cluster 20ghaddar13wang $ROOT"
if [ -f cluster_$SITE.txt ]; then
    [ "$(cat cluster_$SITE.txt)" = "$identity" ] || die 'submission ledger belongs to another cluster/project'
else
    printf '%s\n' "$identity" > cluster_$SITE.txt
fi
protocol_hash="$(cat plan.json code.sha256 | sha256sum | awk '{print $1}')"
queue="$(squeue -h -u "$USER" -o '%A %T %j')" || die 'cannot read your queue'
declare -A active ids complete names
while read -r number state name; do
    if [ -n "$number" ]; then active["$number"]="$state"; names["$name"]="$number"; fi
done <<< "$queue"
submitted=0
while IFS=$'\t' read -r task seconds memory requirements account; do
    case "$account" in aip-bghaddar|aip-boyuwang) ;; *) die "invalid account for $task: $account" ;; esac
    marker="$ROOT/rebuttal_basis_lambda_extra_result/jobs/$task/.done"
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
    if [ -f "$LEDGER" ]; then
        read -r previous old_dependency < <(awk -F '\t' -v task="$task" '$1==task{job=$2;dep=$3} END{print job, dep}' "$LEDGER")
    fi
    if [ -n "$previous" ] && [ -n "${active[$previous]:-}" ]; then
        if [ "${active[$previous]}" = PENDING ] && [ -n "$dependency" ] && [ "$old_dependency" != "$dependency" ]; then
            scontrol update JobId="$previous" Dependency="afterok:$dependency"
            printf '%s\t%s\t%s\t%s\n' "$task" "$previous" "$dependency" "$(date -u +%FT%TZ)" >> "$LEDGER"
            printf 'UPDATED %s: dependencies now %s\n' "$task" "$dependency"
        fi
        ids["$task"]="$previous"
        printf 'SKIP %s: job %s already queued/running\n' "$task" "$previous"
        continue
    fi
    [ -z "${names[bl_$task]:-}" ] || die "task $task is already active outside this ledger: ${names[bl_$task]}"
    arguments=(--parsable --account="$account" --nodes=1 --ntasks=1
        --cpus-per-task=1 --mem="$memory" --gres="$GPU_REQUEST"
        --time="$(printf '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))")"
        --signal=USR1@180 --job-name="bl_$task" --chdir="$ROOT"
        --output="$BATCH/logs/$SITE/%x-%j.out")
    [ -z "$dependency" ] || arguments+=(--dependency="afterok:$dependency")
    number="$(sbatch "${arguments[@]}" "$BATCH/job.sh" "$ROOT" "$BATCH" "$task" "$ENV_ACTIVATE" "$DATA_ROOT" "$SITE")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID: $number"
    ids["$task"]="$number"
    printf '%s\t%s\t%s\t%s\n' "$task" "$number" "${dependency:--}" "$(date -u +%FT%TZ)" >> "$LEDGER"
    printf 'SUBMITTED %s: %s, %s, %ss, %s, %s\n' "$task" "$number" "$account" "$seconds" "$memory" "$GPU_REQUEST"
    submitted=$((submitted+1))
done < <(jq -r --arg site "$SITE" '.tasks[] | select(.site==$site) | [.id,(.seconds|tostring),.memory,(if (.requires|length)>0 then (.requires|join(",")) else "-" end),.account] | @tsv' plan.json)
printf 'Submitted %s jobs on %s. This part has 33 jobs: 20 Ghaddar and 13 Wang, for 12 BASIS cells.\n' "$submitted" "$SITE"
printf 'Rerun this script to resume unfinished work. Combined results: %s/rebuttal_basis_lambda_extra_result/summaries/\n' "$ROOT"
