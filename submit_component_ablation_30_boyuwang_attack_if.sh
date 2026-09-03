#!/usr/bin/env bash
# Submit replacements for only the 30 -M jobs from the component-ablation batch.
# These jobs run from /home/mmoslem3/scratch/attack_if on aip-boyuwang.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MATRIX="$SCRIPT_DIR/sbatch/component_ablation_30/configurations.tsv"
JOB_SCRIPT="$SCRIPT_DIR/sbatch/component_ablation_30_boyuwang_attack_if/job.sh"
LOG_DIR=/home/mmoslem3/scratch/attack_if/sbatch/logs
SOURCE_ROOT=/home/mmoslem3/scratch/attack_if
WALLTIME=0-03:20:00
ACCOUNT=aip-boyuwang
SELECTOR=minus-m

[ -f "$MATRIX" ] || { echo "ERROR: missing $MATRIX" >&2; exit 1; }
[ -f "$JOB_SCRIPT" ] || { echo "ERROR: missing $JOB_SCRIPT" >&2; exit 1; }

config_count=$(awk -F '\t' '!/^#/ && NF { count++ } END { print count + 0 }' "$MATRIX")
[ "$config_count" -eq 30 ] || {
    echo "ERROR: expected 30 configurations in $MATRIX; found $config_count" >&2
    exit 1
}

mkdir -p "$LOG_DIR"
job_count=0

while IFS="$(printf '\t')" read -r config_id model attack class_pair budget; do
    case "$config_id" in
        ''|'#'*) continue ;;
    esac

    model_tag=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')
    attack_tag=$(printf '%s' "$attack" | sed 's/gradmatch/gm/')
    pair_tag=$(printf '%s' "$class_pair" | sed 's/dog-bird/db/; s/frog-airplane/fa/')
    budget_tag=$(printf '%s' "$budget" | tr -d '.')
    job_name="cmp30b_${config_id}_${model_tag}_${attack_tag}_${pair_tag}_b${budget_tag}_minus_m"
    output_path="$LOG_DIR/${job_name}-%j.out"
    export_arg="ALL,SOURCE_ROOT=$SOURCE_ROOT,MODEL=$model,ATTACK=$attack,CLASS_PAIR=$class_pair,BUDGETS=$budget,SELECT=$SELECTOR"

    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'sbatch --account=%s --time=%s --job-name=%s --output=%s --export=%s %s\n' \
            "$ACCOUNT" "$WALLTIME" "$job_name" "$output_path" "$export_arg" "$JOB_SCRIPT"
    else
        sbatch --account="$ACCOUNT" \
            --time="$WALLTIME" \
            --job-name="$job_name" \
            --output="$output_path" \
            --export="$export_arg" \
            "$JOB_SCRIPT"
    fi
    job_count=$((job_count + 1))
done < "$MATRIX"

[ "$job_count" -eq 30 ] || {
    echo "ERROR: expected 30 jobs; prepared $job_count" >&2
    exit 1
}

echo "Prepared $job_count $SELECTOR jobs on $ACCOUNT; walltime $WALLTIME each."
