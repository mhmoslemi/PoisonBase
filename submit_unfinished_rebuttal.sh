#!/bin/bash
# Resume only the seven timed-out W1 tasks and 26 saved defense bundles.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -Eeuo pipefail

ROOT="${DEFENSE_ROOT:-/scratch/mmoslem3/PoisonBase}"
RETRY="$ROOT/sbatch/rebuttal_resume"
W1="$ROOT/sbatch/rebuttal_access_conv_7x8_vulcan"
DEFENSE="$ROOT/sbatch/rebuttal_saved_defenses"
mkdir -p "$RETRY/logs"
exec 9>"$RETRY/submission.lock"
flock -n 9 || { echo 'ERROR: another retry submitter is running.' >&2; exit 1; }
exec 8>"$DEFENSE/submission.lock"
flock -n 8 || { echo 'ERROR: the defense submitter is already running.' >&2; exit 1; }

ACTIVE="$RETRY/.active_jobs.$$"
trap 'rm -f "$ACTIVE"' EXIT
squeue -h -u "$USER" -o '%A' > "$ACTIVE"
submitted=0
while IFS=$'\t' read -r kind index duration memory marker unfinished; do
    [ "$kind" != kind ] || continue
    if [ -s "$ROOT/$marker" ]; then
        echo "SKIP $kind/$index: complete"
        continue
    fi
    if [ "$kind" = w1 ]; then
        ledger="$W1/submissions.tsv"
        worker="$W1/job.sh"
        options=(--export="ALL,ACCESS_ROOT=$ROOT,ACCESS_BATCH=$W1")
        payload=$(awk -F '\t' -v task_id="$index" 'NR>1 && $1==task_id {printf "%s\t%s\t%s\t%s\t%s",$2,$3,$4,$5,$6}' "$W1/manifest.tsv")
        [ -n "$payload" ] || { echo "ERROR: missing W1 task $index" >&2; exit 1; }
    elif [ "$kind" = defense ]; then
        ledger="$DEFENSE/submissions.tsv"
        worker="$RETRY/defense_worker.sh"
        options=(--signal=USR1@180 --export="ALL,DEFENSE_ROOT=$ROOT")
    else
        echo "ERROR: unknown task type $kind" >&2
        exit 1
    fi
    last_job=$(awk -F '\t' -v task_id="$index" '$1==task_id {id=$2} END {print id}' "$ledger")
    if [ -n "$last_job" ] && grep -Fxq "$last_job" "$ACTIVE"; then
        echo "SKIP $kind/$index: job $last_job already queued/running"
        continue
    fi
    job_id=$(sbatch --parsable \
        --account=aip-bghaddar --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --mem="${memory}G" --gres=gpu:l40s:1 --time="$duration" \
        --job-name="reb_retry_${kind}_${index}" --chdir="$ROOT" \
        --output="$RETRY/logs/%x-%j.out" \
        "${options[@]}" "$worker" "$index")
    job_id="${job_id%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid job ID $job_id" >&2; exit 1; }
    if [ "$kind" = w1 ]; then
        printf '%s\t%s\t%s\n' "$index" "$job_id" "$payload" >> "$ledger"
    else
        printf '%s\t%s\tnone\t%s\n' "$index" "$job_id" "$(date -u +%FT%TZ)" >> "$ledger"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$kind" "$index" "$job_id" "$duration" "$(date -u +%FT%TZ)" >> "$RETRY/submissions.tsv"
    submitted=$((submitted+1))
    echo "SUBMITTED $kind/$index job=$job_id time=$duration memory=${memory}G unfinished_victims=$unfinished"
done < "$RETRY/jobs.tsv"
echo "Submitted $submitted retry jobs. Completed results are retained; defense checkpoints resume."
