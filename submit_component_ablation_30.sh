#!/usr/bin/env bash
# Submit the same 30 configurations for six component selectors: 180 jobs.
# All 30 -M jobs use aip-boyuwang; the other 150 use aip-yiweilu.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BATCH_DIR="$SCRIPT_DIR/sbatch/component_ablation_30"
MATRIX="$BATCH_DIR/configurations.tsv"
JOB_SCRIPT="$BATCH_DIR/job.sh"
LOG_DIR="$SCRIPT_DIR/sbatch/logs"
SOURCE_ROOT=/home/mmoslem3/scratch/PoisonBase
WALLTIME=0-03:20:00
SELECTORS='minus-m r a a-minus-m a-plus-r minus-m-times-r'

[ -f "$MATRIX" ] || { echo "ERROR: missing $MATRIX" >&2; exit 1; }
[ -f "$JOB_SCRIPT" ] || { echo "ERROR: missing $JOB_SCRIPT" >&2; exit 1; }

config_count=$(awk -F '\t' '!/^#/ && NF { count++ } END { print count + 0 }' "$MATRIX")
[ "$config_count" -eq 30 ] || {
    echo "ERROR: expected 30 configurations in $MATRIX; found $config_count" >&2
    exit 1
}

mkdir -p "$LOG_DIR"
job_count=0
boyuwang_count=0
yiweilu_count=0

while IFS="$(printf '\t')" read -r config_id model attack class_pair budget; do
    case "$config_id" in
        ''|'#'*) continue ;;
    esac

    for selection in $SELECTORS; do
        case "$selection" in
            minus-m) account=aip-boyuwang; selector_tag=minus_m ;;
            r) account=aip-yiweilu; selector_tag=r ;;
            a) account=aip-yiweilu; selector_tag=a ;;
            a-minus-m) account=aip-yiweilu; selector_tag=a_minus_m ;;
            a-plus-r) account=aip-yiweilu; selector_tag=a_plus_r ;;
            minus-m-times-r) account=aip-yiweilu; selector_tag=minus_m_times_r ;;
        esac

        if [ "$account" = aip-boyuwang ]; then
            boyuwang_count=$((boyuwang_count + 1))
        else
            yiweilu_count=$((yiweilu_count + 1))
        fi

        model_tag=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')
        attack_tag=$(printf '%s' "$attack" | sed 's/gradmatch/gm/')
        pair_tag=$(printf '%s' "$class_pair" | sed 's/dog-bird/db/; s/frog-airplane/fa/')
        budget_tag=$(printf '%s' "$budget" | tr -d '.')
        job_name="cmp30_${config_id}_${model_tag}_${attack_tag}_${pair_tag}_b${budget_tag}_${selector_tag}"
        output_path="$LOG_DIR/${job_name}-%j.out"
        export_arg="ALL,SOURCE_ROOT=$SOURCE_ROOT,MODEL=$model,ATTACK=$attack,CLASS_PAIR=$class_pair,BUDGETS=$budget,SELECT=$selection"

        if [ "${DRY_RUN:-0}" = 1 ]; then
            printf 'sbatch --account=%s --time=%s --job-name=%s --output=%s --export=%s %s\n' \
                "$account" "$WALLTIME" "$job_name" "$output_path" "$export_arg" "$JOB_SCRIPT"
        else
            sbatch --account="$account" \
                --time="$WALLTIME" \
                --job-name="$job_name" \
                --output="$output_path" \
                --export="$export_arg" \
                "$JOB_SCRIPT"
        fi
        job_count=$((job_count + 1))
    done
done < "$MATRIX"

[ "$job_count" -eq 180 ] || {
    echo "ERROR: expected 180 jobs; prepared $job_count" >&2
    exit 1
}
[ "$boyuwang_count" -eq 30 ] || {
    echo "ERROR: expected 30 aip-boyuwang jobs; prepared $boyuwang_count" >&2
    exit 1
}
[ "$yiweilu_count" -eq 150 ] || {
    echo "ERROR: expected 150 aip-yiweilu jobs; prepared $yiweilu_count" >&2
    exit 1
}

echo "Prepared $job_count jobs: $boyuwang_count on aip-boyuwang and $yiweilu_count on aip-yiweilu; walltime $WALLTIME each."
