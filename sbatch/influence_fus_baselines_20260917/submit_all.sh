#!/usr/bin/env bash
# Submit Gao preprocessing, 24 static-selector jobs, and eight four-part FUS arrays.
set -euo pipefail

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conv_raw="$(sbatch --parsable "$job_dir/precompute_metrics_conv.sh")"
r20_raw="$(sbatch --parsable "$job_dir/precompute_metrics_r20.sh")"
conv_metrics="${conv_raw%%;*}"
r20_metrics="${r20_raw%%;*}"
printf 'submitted ConvNet metric array: %s\n' "$conv_metrics"
printf 'submitted ResNet20 metric array: %s\n' "$r20_metrics"

config_count=0
gpu_task_count=0
merge_count=0
for job in "$job_dir"/job_*.sh; do
    selector="$(sed -n 's/^export IFB_SELECTOR=//p' "$job")"
    model="$(sed -n 's/^export IFB_MODEL=//p' "$job")"
    if [[ "$selector" == gao-* ]]; then
        case "$model" in
            ConvNetBN) dependency="$conv_metrics" ;;
            ResNet20BN) dependency="$r20_metrics" ;;
            *) printf 'unknown model in %s: %s\n' "$job" "$model" >&2; exit 1 ;;
        esac
        raw="$(sbatch --parsable --dependency="afterok:$dependency" "$job")"
        job_id="${raw%%;*}"
        printf 'submitted static config %s: %s\n' "$(basename "$job")" "$job_id"
        gpu_task_count=$((gpu_task_count + 1))
    else
        raw="$(sbatch --parsable "$job")"
        job_id="${raw%%;*}"
        merge_job="$job_dir/$(basename "${job/job_/merge_}")"
        [ -f "$merge_job" ] || {
            printf 'missing FUS merge job: %s\n' "$merge_job" >&2
            exit 1
        }
        merge_raw="$(sbatch --parsable --dependency="afterok:$job_id" \
            "$merge_job")"
        merge_id="${merge_raw%%;*}"
        printf 'submitted FUS array %s: %s; merge: %s\n' \
            "$(basename "$job")" "$job_id" "$merge_id"
        gpu_task_count=$((gpu_task_count + 4))
        merge_count=$((merge_count + 1))
    fi
    config_count=$((config_count + 1))
done
printf 'submitted %d configurations as %d GPU tasks, plus %d dependent merge jobs\n' \
    "$config_count" "$gpu_task_count" "$merge_count"
