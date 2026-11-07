#!/bin/bash
# Submit only the ConvNet cells required by the first rebuttal table.
set -Eeuo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
BATCH="$ROOT/sbatch/rebuttal_access_conv_5x4_fir"
JOB_SCRIPT="$ROOT/experiments/rebuttal_access_conv/job.sh"
ACCOUNT=def-bghaddar

command -v sbatch >/dev/null 2>&1 || {
    echo "ERROR: run this script on the Fir login node" >&2
    exit 1
}
[[ -f "$JOB_SCRIPT" ]] || { echo "ERROR: missing $JOB_SCRIPT" >&2; exit 1; }
[[ -d "$ROOT/data/cifar-10-batches-py" ]] || { echo "ERROR: missing CIFAR-10 data" >&2; exit 1; }

# Verify the full-data pool has the 30 checkpoints needed for three overlapping
# 20-model ensembles, and every partial ConvNet pool has 20 checkpoints.
full="$ROOT/cache/surrogates/ConvNetBN_60ep_lr0.1_bs128_seed42"
full_count=$(find "$full" -maxdepth 1 -type f -name 'net_*.pt' -printf '.' | wc -c)
[[ "$full_count" -ge 30 ]] || { echo "ERROR: full ConvNet pool has $full_count/30 checkpoints" >&2; exit 1; }
for fraction in 0.2 0.5; do
    for draw in 0 1 2; do
        matches=("$ROOT"/reviewer_20261106_fir/reviewer_20261106_result/surrogates/ConvNetBN_f${fraction}_r${draw}_*)
        [[ ${#matches[@]} -eq 1 && -d "${matches[0]}" ]] || {
            echo "ERROR: expected one ConvNet f=$fraction r=$draw pool" >&2
            exit 1
        }
        count=$(find "${matches[0]}" -maxdepth 1 -type f -name 'net_*.pt' -printf '.' | wc -c)
        [[ "$count" -eq 20 ]] || {
            echo "ERROR: ${matches[0]} has $count/20 checkpoints" >&2
            exit 1
        }
    done
done

mkdir -p "$BATCH/logs"
LEDGER="$BATCH/submissions.tsv"
touch "$LEDGER"
exec 9>"$BATCH/submit.lock"
flock -n 9 || { echo "ERROR: another submitter is active" >&2; exit 1; }

submit_one() {
    local label="$1" duration="$2"
    shift 2
    local old job_id
    old=$(awk -F '\t' -v label="$label" '$1==label {print $2}' "$LEDGER")
    if [[ -n "$old" ]]; then
        echo "Existing $label: $old"
        return
    fi
    job_id=$(sbatch --parsable \
        --account="$ACCOUNT" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G \
        --gpus-per-node=h100:1 --time="$duration" --job-name="reb_${label}" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        --export="ALL,ACCESS_ROOT=$ROOT" "$JOB_SCRIPT" "$@")
    job_id=${job_id%%;*}
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch response: $job_id" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\n' "$label" "$job_id" "$duration" "$*" >> "$LEDGER"
    echo "Submitted $label: $job_id"
}

# Four jobs, each containing two Random table cells.
for pair in 0 1 2 3; do
    submit_one "rand_${pair}" 03:00:00 --mode random-pair --index "$pair"
done

# Sixteen jobs: one separate M job and one separate BASIS job for each condition.
for method in margin basis; do
    for condition in 0 1 2 3 4 5 6 7; do
        submit_one "${method}_${condition}" 04:00:00 \
            --mode method-cell --method "$method" --index "$condition"
    done
done

echo "Submitted/recorded 20 jobs. Ledger: $LEDGER"
