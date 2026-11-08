#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT="$(cd -- "${SOURCE_ROOT:-$SCRIPT_DIR}" && pwd -P)"
BATCH="$SCRIPT_DIR/sbatch/rebuttal_mr_killarney"
WRAPPER="$SCRIPT_DIR/sbatch/rebuttal_mr_vulcan_victims"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for program in sbatch squeue scontrol sinfo jq flock sha256sum rsync; do command -v "$program" >/dev/null || die "missing command: $program"; done
[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
[ -s "$ROOT/data/tinyimagenet.pt" ] || die 'missing data/tinyimagenet.pt'
cd "$BATCH"
sha256sum --check --quiet code.sha256 || die 'the original frozen M/R snapshot changed'
protocol_hash="$(cat plan.json code.sha256 | sha256sum | awk '{print $1}')"
jq -e --arg hash "$protocol_hash" '.complete and .protocol_hash==$hash' "$ROOT/rebuttal_mr_result/preflight.json" >/dev/null || die 'missing or incompatible original preflight'
tasks="$(jq -c '.tasks[] | select(.stage=="victims" and (.cell=="tiny_gm_M" or .cell=="tiny_gm_R"))' plan.json)"
[ "$(printf '%s\n' "$tasks" | wc -l)" -eq 8 ] || die 'expected exactly eight victim-only tasks'
files=()
while IFS= read -r record; do
    cell="$(jq -r '.cell' <<< "$record")"
    target="$(jq -r '.targets[0]' <<< "$record")"
    relative="rebuttal_mr_result/cells/$cell/target_$target/poison.pt"
    [ -s "$ROOT/$relative" ] || die "missing completed poison: $ROOT/$relative"
    files+=("$relative")
done <<< "$tasks"
cluster="$(scontrol show config | awk '$1=="ClusterName"{print $3}')"
[[ "${cluster,,}" == *vulcan* ]] || die 'run this script on Vulcan'
inventory="$(sinfo -h -o '%G' --Node)"
[[ "$inventory" == *gpu:l40s:* ]] || die 'no L40S GPU available'
mkdir -p "$WRAPPER/logs"
exec 9>"$WRAPPER/submission.lock"
flock -n 9 || die 'another victim submitter is running'
if [ -f "$WRAPPER/poisons.sha256" ]; then
    (cd "$ROOT" && sha256sum --check --quiet "$WRAPPER/poisons.sha256") || die 'poison files changed'
else
    (cd "$ROOT" && sha256sum "${files[@]}") > "$WRAPPER/poisons.sha256"
fi
queue="$(squeue -h -u "$USER" -o '%A %T %j')"
declare -A active names
while read -r number state name; do
    if [ -n "$number" ]; then active["$number"]="$state"; [ -z "$name" ] || names["$name"]="$number"; fi
done <<< "$queue"
submitted=0
while IFS= read -r record; do
    task="$(jq -r '.id' <<< "$record")"
    marker="$ROOT/rebuttal_mr_result/jobs/$task/.done"
    if [ -f "$marker" ]; then
        jq -e --arg hash "$protocol_hash" '.protocol_hash==$hash' "$marker" >/dev/null || die "incompatible completed task: $task"
        printf 'SKIP %s: complete\n' "$task"; continue
    fi
    previous=''
    if [ -f "$WRAPPER/submissions.tsv" ]; then previous="$(awk -F '\t' -v task="$task" '$1==task{job=$2} END{print job}' "$WRAPPER/submissions.tsv")"; fi
    if [ -n "$previous" ] && [ -n "${active[$previous]:-}" ]; then printf 'SKIP %s: job %s active\n' "$task" "$previous"; continue; fi
    [ -z "${names[mrv_$task]:-}" ] || die "task already active: $task"
    number="$(sbatch --parsable --account=aip-bghaddar --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=18G --gres=gpu:l40s:1 --time=04:00:00 \
        --signal=USR1@180 --job-name="mrv_$task" --chdir="$ROOT" \
        --output="$WRAPPER/logs/%x-%j.out" "$WRAPPER/job.sh" "$ROOT" "$task" "$ENV_ACTIVATE")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid job ID: $number"
    printf '%s\t%s\t%s\n' "$task" "$number" "$(date -u +%FT%TZ)" >> "$WRAPPER/submissions.tsv"
    printf 'SUBMITTED %s: %s, aip-bghaddar, 4h, 18G, one L40S\n' "$task" "$number"
    submitted=$((submitted+1))
done <<< "$tasks"
printf 'Submitted %s victim-only jobs. No selection or crafting.\n' "$submitted"
