#!/usr/bin/env bash
# Submit the 11 M-related BASIS anomaly reruns from the PoisonBase checkout.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MATRIX="$SCRIPT_DIR/configurations.tsv"
JOB_SCRIPT="$SCRIPT_DIR/job.sh"
SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
PERSIST_DATA_ROOT="$SOURCE_ROOT/data"
LOG_DIR="$SOURCE_ROOT/sbatch/logs"
ACCOUNT=aip-yiweilu
WALLTIME=0-03:30:00

[ -f "$MATRIX" ] || { echo "ERROR: missing $MATRIX" >&2; exit 1; }
[ -f "$JOB_SCRIPT" ] || { echo "ERROR: missing $JOB_SCRIPT" >&2; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$LOG_DIR"

job_count=0
while IFS="$(printf '\t')" read -r config_id model attack class_pair budget lambda_margin violation table_source basis_asr r_asr m_asr; do
    case "$config_id" in ''|'#'*) continue ;; esac

    model_tag=$(printf '%s' "$model" | sed 's/ConvNetBN/conv/; s/ResNet20BN/r20/; s/VGG13BN/vgg/')
    attack_tag=$(printf '%s' "$attack" | sed 's/gradmatch/gm/; s/fc/bp/')
    pair_tag=$(printf '%s' "$class_pair" | sed 's/dog-bird/db/; s/frog-airplane/fa/')
    budget_tag=$(printf '%s' "$budget" | tr -d '.')
    job_name="bm70${config_id}_${model_tag}_${attack_tag}_${pair_tag}_b${budget_tag}_l${lambda_margin}"
    output_path="$LOG_DIR/${job_name}-%j.out"
    export_arg="ALL,SOURCE_ROOT=$SOURCE_ROOT,PERSIST_DATA_ROOT=$PERSIST_DATA_ROOT,PYTHON_ENV=/home/mmoslem3/ENV,ENV_ACTIVATE=/home/mmoslem3/ENV/bin/activate,MODEL=$model,ATTACK=$attack,CLASS_PAIR=$class_pair,BUDGETS=$budget,LAMBDA_MARGIN=$lambda_margin,SELECT=ours,BASE_DIST=cosine_norm,VICTIM_EPOCHS=70,VICTIM_DECAY=50"

    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'sbatch --account=%s --time=%s --job-name=%s --output=%s --export=%s %s\n' \
            "$ACCOUNT" "$WALLTIME" "$job_name" "$output_path" "$export_arg" "$JOB_SCRIPT"
    else
        sbatch --account="$ACCOUNT" --time="$WALLTIME" \
            --job-name="$job_name" --output="$output_path" \
            --export="$export_arg" "$JOB_SCRIPT"
    fi
    job_count=$((job_count + 1))
done < "$MATRIX"

[ "$job_count" -eq 11 ] || {
    echo "ERROR: expected 11 jobs; prepared $job_count" >&2
    exit 1
}
echo "Prepared $job_count PoisonBase jobs on $ACCOUNT; walltime $WALLTIME each."
