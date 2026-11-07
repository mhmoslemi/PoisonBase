#!/bin/bash
# Submit the 16 missing Wicked Oddities cells in the Part 2 rebuttal table.
set -Eeuo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
BATCH="$ROOT/sbatch/rebuttal_wo_table_fir"
JOB_SCRIPT="$ROOT/experiments/rebuttal_wo_table/job.sh"
ACCOUNT=def-bghaddar

command -v sbatch >/dev/null 2>&1 || {
    echo "ERROR: run this script on the Fir login node" >&2
    exit 1
}
[[ -f "$JOB_SCRIPT" ]] || { echo "ERROR: missing $JOB_SCRIPT" >&2; exit 1; }
[[ -d "$ROOT/data/cifar-10-batches-py" ]] || { echo "ERROR: missing CIFAR-10 data" >&2; exit 1; }
[[ -f "$ROOT/reviewer_20261106_result/wicked/wo_pretrained.pt" ]] || {
    echo "ERROR: missing completed WO-KNN ranking" >&2; exit 1;
}
[[ -f "$ROOT/reviewer_20261106_fir/reviewer_20261106_result/wicked/wo_ood.pt" ]] || {
    echo "ERROR: missing completed WO-OOD ranking" >&2; exit 1;
}
for model in ConvNetBN ResNet20BN; do
    pool="$ROOT/cache/surrogates/${model}_60ep_lr0.1_bs128_seed42"
    count=$(find "$pool" -maxdepth 1 -type f -name 'net_*.pt' -printf '.' | wc -c)
    [[ "$count" -ge 20 ]] || {
        echo "ERROR: $model surrogate pool has $count/20 checkpoints" >&2
        exit 1
    }
done

mkdir -p "$BATCH/logs"
LEDGER="$BATCH/submissions.tsv"
touch "$LEDGER"
exec 9>"$BATCH/submit.lock"
flock -n 9 || { echo "ERROR: another submitter is active" >&2; exit 1; }

LAST_JOB_ID=
submit_one() {
    local label="$1" duration="$2" gpu="$3" dependency="$4" mode="$5"
    local old job_id
    old=$(awk -F '\t' -v label="$label" '$1==label {print $2}' "$LEDGER")
    if [[ -n "$old" ]]; then
        LAST_JOB_ID="$old"
        echo "Existing $label: $old"
        return
    fi
    options=(
        --parsable --account="$ACCOUNT" --nodes=1 --ntasks=1
        --cpus-per-task=1 --mem=6G --time="$duration"
        --job-name="rebwo_${label}" --chdir="$ROOT"
        --output="$BATCH/logs/%x-%j.out" --export="ALL,WO_ROOT=$ROOT"
    )
    [[ "$gpu" == yes ]] && options+=(--gpus-per-node=h100:1)
    [[ -n "$dependency" ]] && options+=(--dependency="afterok:$dependency")
    job_id=$(sbatch "${options[@]}" "$JOB_SCRIPT" "$mode")
    job_id=${job_id%%;*}
    [[ "$job_id" =~ ^[0-9]+$ ]] || {
        echo "ERROR: invalid sbatch response: $job_id" >&2
        exit 1
    }
    printf '%s\t%s\t%s\t%s\n' "$label" "$job_id" "$duration" "$mode" >> "$LEDGER"
    LAST_JOB_ID="$job_id"
    echo "Submitted $label: $job_id"
}

# The short CPU job stages the two already-computed WO rankings into this run.
submit_one prep 00:15:00 no "" prep
prep_id="$LAST_JOB_ID"

# Four GPU jobs cover all 16 cells; every victim trains for 50 epochs.
submit_one conv_all 06:00:00 yes "$prep_id" conv-all
submit_one res_b002 06:00:00 yes "$prep_id" res-b002
submit_one res_b005_knn 06:00:00 yes "$prep_id" res-b005-knn
submit_one res_b005_ood 06:00:00 yes "$prep_id" res-b005-ood

echo "Submitted/recorded 5 jobs. Ledger: $LEDGER"
