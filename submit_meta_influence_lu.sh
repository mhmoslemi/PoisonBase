#!/bin/bash
# Prepare/submission entry point for only the two missing Reviewer 3 rows.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_meta_influence"
PLAN="$BATCH/plan.json"
RESULTS="$ROOT/rebuttal_meta_influence_result"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
for program in sbatch squeue scontrol jq flock rsync sha256sum; do
    command -v "$program" >/dev/null || die "missing command: $program"
done
[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
[ -d "$ROOT/data/cifar-10-batches-py" ] || die 'missing CIFAR-10 data'
cd "$BATCH"
sha256sum --check --quiet code.sha256 || die 'the frozen experiment code has changed'
for model in ConvNetBN ResNet20BN; do
    for ((index=0; index<20; index++)); do
        path="$ROOT/cache/surrogates/${model}_60ep_lr0.1_bs128_seed42/net_$index.pt"
        [ -s "$path" ] || die "missing surrogate: $path"
    done
    for ((index=0; index<6; index++)); do
        path="$ROOT/cache/clean_victims/${model}_50ep_lr0.1_bs125_wd0_seed42/net_$index.pt"
        [ -s "$path" ] || die "missing clean victim: $path"
    done
    for family in bp gm; do
        file_tag=fc; [ "$family" != gm ] || file_tag=gradmatch
        target_file="$ROOT/target_sets/${model}_${file_tag}_dog-bird.json"
        got="$(jq -c '.pairs["dog-bird"].indices[:8]' "$target_file")"
        expected="$(jq -c --arg model "$model" --arg family "$family" '.targets[$model][$family]' "$PLAN")"
        [ "$got" = "$expected" ] || die "BP/GM target list differs from the pinned plan: $target_file"
    done
done

mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another copy of this submitter is running'
declare -A live jobs
while IFS='|' read -r number state; do
    [ -n "$number" ] && live["$number"]="$state"
done < <(squeue -h -u "$USER" -o '%i|%T')
submitted=0
position=0
account_cycle=(
    aip-bghaddar aip-bghaddar aip-boyuwang aip-bghaddar aip-bghaddar
    aip-yiweilu aip-bghaddar aip-bghaddar aip-boyuwang aip-bghaddar
)
while IFS= read -r record; do
    id="$(jq -r '.id' <<< "$record")"
    account="${SUBMIT_ACCOUNT:-${account_cycle[$((position % 10))]}}"
    position=$((position+1))
    if [ -f "$RESULTS/jobs/$id/.done" ]; then
        jobs["$id"]=''
        printf 'SKIP %s: complete\n' "$id"
        continue
    fi
    dependency=''
    while IFS= read -r parent; do
        [ -n "$parent" ] || continue
        parent_job="${jobs[$parent]:-}"
        if [ -n "$parent_job" ]; then
            dependency="${dependency:+$dependency:}$parent_job"
        elif [ ! -f "$RESULTS/jobs/$parent/.done" ]; then
            die "dependency $parent has neither a completed marker nor a submitted job"
        fi
    done < <(jq -r '.dependencies[]' <<< "$record")
    last_job=''; old_dependency=''
    if [ -f "$BATCH/submissions.tsv" ]; then
        last_record="$(awk -F '\t' -v task="$id" '$1==task{entry=$2 "|" $3} END{print entry}' "$BATCH/submissions.tsv")"
        IFS='|' read -r last_job old_dependency <<< "$last_record"
    fi
    if [ -n "$last_job" ] && [ -n "${live[$last_job]:-}" ]; then
        jobs["$id"]="$last_job"
        # Reconnect only this batch's queued dependents to resubmitted scorers.
        if [ "${live[$last_job]}" = PENDING ] && [ "${old_dependency:-none}" != "${dependency:-none}" ]; then
            scontrol update JobId="$last_job" Dependency="${dependency:+afterok:$dependency}"
            printf '%s\t%s\t%s\t%s\n' "$id" "$last_job" "${dependency:-none}" "$(date -u +%FT%TZ)" >> "$BATCH/submissions.tsv"
        fi
        printf 'SKIP %s: job %s already queued/running\n' "$id" "$last_job"
        continue
    fi
    duration="$(jq -r '.time' <<< "$record")"
    flags=()
    [ -z "$dependency" ] || flags+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    number="$(sbatch --parsable --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$duration" --signal=B:USR1@180 --job-name="mi_$id" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        "${flags[@]}" "$BATCH/job.sh" "$ROOT" "$id" "$ENV_ACTIVATE")"
    number="${number%%;*}"
    [[ "$number" =~ ^[0-9]+$ ]] || die "invalid job ID: $number"
    jobs["$id"]="$number"
    printf '%s\t%s\t%s\t%s\n' "$id" "$number" "${dependency:-none}" "$(date -u +%FT%TZ)" >> "$BATCH/submissions.tsv"
    printf 'SUBMITTED %s: job=%s time=%s dependency=%s account=%s\n' "$id" "$number" "$duration" "${dependency:-none}" "$account"
    submitted=$((submitted+1))
done < <(jq -c '.tasks[]' "$PLAN")
printf 'Submitted %s jobs. Plan: 14 scoring + 24 cell evaluations = 38 jobs.\n' "$submitted"
if [ -n "${SUBMIT_ACCOUNT:-}" ]; then
    printf 'Account for new submissions: %s.\n' "$SUBMIT_ACCOUNT"
else
    printf 'Full-batch account split: 27 aip-bghaddar, 7 aip-boyuwang, 4 aip-yiweilu.\n'
fi
printf 'Rerun this script to resume unfinished work. Results: %s\n' "$RESULTS"
