#!/usr/bin/env bash
# Submit 20 new configurations x {exact, a-mr} = 40 independent Slurm jobs.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BATCH_DIR="$SCRIPT_DIR/sbatch/exact_amr_batch2"
MATRIX="$BATCH_DIR/configurations.tsv"
JOB_SCRIPT="$BATCH_DIR/job.sh"
LOG_DIR="$SCRIPT_DIR/sbatch/logs"
SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase

[ -f "$MATRIX" ] || { echo "ERROR: missing $MATRIX" >&2; exit 1; }
[ -f "$JOB_SCRIPT" ] || { echo "ERROR: missing $JOB_SCRIPT" >&2; exit 1; }

config_count=$(awk -F '\t' '!/^#/ && NF { count++ } END { print count + 0 }' "$MATRIX")
[ "$config_count" -eq 20 ] || {
    echo "ERROR: expected 20 configurations in $MATRIX; found $config_count" >&2
    exit 1
}

mkdir -p "$LOG_DIR"
job_count=0

while IFS="$(printf '\t')" read -r config_id model attack class_pair budget; do
    case "$config_id" in
        ''|'#'*) continue ;;
    esac

    for selection in exact a-mr; do
        case "$selection" in
            exact) selector_tag=exact ;;
            a-mr)  selector_tag=a_mr ;;
        esac
        model_tag=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')
        attack_tag=$(printf '%s' "$attack" | sed 's/gradmatch/gm/')
        pair_tag=$(printf '%s' "$class_pair" | sed 's/dog-bird/db/; s/frog-airplane/fa/')
        budget_tag=$(printf '%s' "$budget" | tr -d '.')
        job_name="xamr2_${config_id}_${model_tag}_${attack_tag}_${pair_tag}_b${budget_tag}_${selector_tag}"
        output_path="$LOG_DIR/${job_name}-boyuwang-%j.out"
        export_arg="ALL,SOURCE_ROOT=$SOURCE_ROOT,MODEL=$model,ATTACK=$attack,CLASS_PAIR=$class_pair,BUDGETS=$budget,SELECT=$selection"

        if [ "${DRY_RUN:-0}" = 1 ]; then
            printf 'sbatch --account=%s --job-name=%s --output=%s --export=%s %s\n' \
                aip-yiweilu "$job_name" "$output_path" "$export_arg" "$JOB_SCRIPT"
        else
            sbatch --account=aip-boyuwang \
                --job-name="$job_name" \
                --output="$output_path" \
                --export="$export_arg" \
                "$JOB_SCRIPT"
        fi
        job_count=$((job_count + 1))
    done
done < "$MATRIX"

[ "$job_count" -eq 40 ] || {
    echo "ERROR: expected to submit 40 jobs; prepared $job_count" >&2
    exit 1
}
