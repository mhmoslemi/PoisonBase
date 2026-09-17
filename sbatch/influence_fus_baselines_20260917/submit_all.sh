#!/usr/bin/env bash
# Submit the two Gao metric arrays, then the 32 experiment configurations.
set -euo pipefail

job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conv_raw="$(sbatch --parsable "$job_dir/precompute_metrics_conv.sh")"
r20_raw="$(sbatch --parsable "$job_dir/precompute_metrics_r20.sh")"
conv_metrics="${conv_raw%%;*}"
r20_metrics="${r20_raw%%;*}"
printf 'submitted ConvNet metric array: %s\n' "$conv_metrics"
printf 'submitted ResNet20 metric array: %s\n' "$r20_metrics"

count=0
for job in "$job_dir"/job_*.sh; do
    selector="$(sed -n 's/^export IFB_SELECTOR=//p' "$job")"
    model="$(sed -n 's/^export IFB_MODEL=//p' "$job")"
    if [[ "$selector" == gao-* ]]; then
        case "$model" in
            ConvNetBN) dependency="$conv_metrics" ;;
            ResNet20BN) dependency="$r20_metrics" ;;
            *) printf 'unknown model in %s: %s\n' "$job" "$model" >&2; exit 1 ;;
        esac
        sbatch --dependency="afterok:$dependency" "$job"
    else
        sbatch "$job"
    fi
    count=$((count + 1))
done
printf 'submitted %d experiment jobs (Gao jobs depend on their metric array)\n' "$count"
