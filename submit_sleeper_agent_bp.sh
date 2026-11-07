#!/bin/bash
# Submit only the four missing Reviewer 3 gradient-norm BP cells on Vulcan.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BATCH="$ROOT/sbatch/rebuttal_sleeper_agent_bp"
RESULTS="$ROOT/rebuttal_sleeper_agent_bp_result"
ENV_ACTIVATE="${ENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$ENV_ACTIVATE" ] || die "missing environment: $ENV_ACTIVATE"
[ -d "$ROOT/data/cifar-10-batches-py" ] || die 'missing CIFAR-10 data'
for source_file in final_update.py networks.py utils.py sbatch/rebuttal_sleeper_agent_bp/job.sh; do
    [ -s "$ROOT/$source_file" ] || die "missing $source_file"
done
for model in ConvNetBN ResNet20BN; do
    [ -s "$ROOT/target_sets/${model}_fc_dog-bird.json" ] || die "missing BP targets for $model"
    for ((index=0; index<20; index++)); do
        for cached_file in \
            "surrogates/${model}_60ep_lr0.1_bs128_seed42/net_$index.pt" \
            "selector_metrics/CIFAR10_${model}_60ep_select10_lr0.1_bs128_decay35-45_wd0_seed42/class5/net_$index.npz"; do
            [ -s "$ROOT/cache/$cached_file" ] || die "missing $ROOT/cache/$cached_file"
        done
    done
    for ((index=0; index<6; index++)); do
        cached_file="clean_victims/${model}_50ep_lr0.1_bs125_wd0_seed42/net_$index.pt"
        [ -s "$ROOT/cache/$cached_file" ] || die "missing $ROOT/cache/$cached_file"
    done
done

mkdir -p "$BATCH/logs"
exec 9>"$BATCH/submission.lock"
flock -n 9 || die 'another copy of this submitter is running'
active_names="$(squeue -h -u "$USER" -o '%j')"
submitted=0
for model in ConvNetBN ResNet20BN; do
    if [ "$model" = ConvNetBN ]; then model_tag=conv; else model_tag=res; fi
    for budget in 0.002 0.005; do
        job_name="rebsa_${model_tag}_b${budget#0.}"
        if [ -f "$RESULTS/${model}_b${budget}/.done" ]; then
            printf 'SKIP %s: complete\n' "$job_name"
            continue
        fi
        if printf '%s\n' "$active_names" | grep -Fxq "$job_name"; then
            printf 'SKIP %s: already queued/running\n' "$job_name"
            continue
        fi
        job_id="$(sbatch --parsable \
            --account=aip-bghaddar --nodes=1 --ntasks=1 --cpus-per-task=1 \
            --mem=6G --gres=gpu:l40s:1 --time=04:00:00 \
            --signal=B:USR1@180 --job-name="$job_name" --chdir="$ROOT" \
            --output="$BATCH/logs/%x-%j.out" \
            "$BATCH/job.sh" "$ROOT" "$model" "$budget" "$ENV_ACTIVATE")"
        job_id="${job_id%%;*}"
        [[ "$job_id" =~ ^[0-9]+$ ]] || die "invalid job ID: $job_id"
        printf '%s\t%s\t%s\t%s\n' "$model" "$budget" "$job_id" "$(date -u +%FT%TZ)" >> "$BATCH/submissions.tsv"
        printf 'SUBMITTED %s: job=%s, 4h, 1 CPU, 6G, 1 L40S, 8 targets x 6 victims\n' "$job_name" "$job_id"
        submitted=$((submitted+1))
    done
done
printf 'Submitted %s jobs. Rerun this script to resume unfinished cells.\n' "$submitted"
