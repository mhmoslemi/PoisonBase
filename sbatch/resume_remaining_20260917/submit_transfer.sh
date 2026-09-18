#!/usr/bin/env bash
# Resume only the 16 incomplete fresh architecture-transfer configurations.
set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
SOURCE_DIR="$ROOT/sbatch/victim_transfer_fresh_20260917"
RESULT_ROOT="${VXF_RESULT_ROOT:-$ROOT/victim_transfer_fresh_20260917_result}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COUNT="$HERE/count_results.py"
export ROOT RESULT_ROOT

submit_sbatch() {
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf 'DRY RUN:' >&2
        printf ' %q' sbatch "$@" >&2
        printf '\n' >&2
        printf 'DRYRUN\n'
    else
        sbatch "$@"
    fi
}

jobs=(
    job_023_gm_b0005_k10_ar20_vr20.sh
    job_024_gm_b0005_k10_ar20_vvgg.sh
    job_026_gm_b0005_k10_avgg_vr20.sh
    job_027_gm_b0005_k10_avgg_vvgg.sh
    job_032_gm_b0005_k20_ar20_vr20.sh
    job_033_gm_b0005_k20_ar20_vvgg.sh
    job_035_gm_b0005_k20_avgg_vr20.sh
    job_036_gm_b0005_k20_avgg_vvgg.sh
    job_059_sapa_b0005_k10_ar20_vr20.sh
    job_060_sapa_b0005_k10_ar20_vvgg.sh
    job_062_sapa_b0005_k10_avgg_vr20.sh
    job_063_sapa_b0005_k10_avgg_vvgg.sh
    job_068_sapa_b0005_k20_ar20_vr20.sh
    job_069_sapa_b0005_k20_ar20_vvgg.sh
    job_071_sapa_b0005_k20_avgg_vr20.sh
    job_072_sapa_b0005_k20_avgg_vvgg.sh
)

submitted=0
skipped=0
for name in "${jobs[@]}"; do
    job="$SOURCE_DIR/$name"
    [ -f "$job" ] || { printf 'missing job file: %s\n' "$job" >&2; exit 1; }
    run_name="$(sed -n 's/^export VXF_RUN_NAME=//p' "$job")"
    [ -n "$run_name" ] || { printf 'missing VXF_RUN_NAME in %s\n' "$job" >&2; exit 1; }
    complete="$(python3 "$COUNT" "$RESULT_ROOT/$run_name")"
    if [ "$complete" -ge 60 ]; then
        printf 'skip complete transfer config: %s (60/60)\n' "$name"
        skipped=$((skipped + 1))
        continue
    fi

    old_job_name="$(sed -n 's/^#SBATCH --job-name=//p' "$job")"
    resume_job_name="${old_job_name/vxf/vxr}"
    raw="$(submit_sbatch --parsable \
        --job-name="$resume_job_name" \
        --output="$ROOT/sbatch/logs/${resume_job_name}-%j.out" \
        --export=ALL,VXF_RESUME=1 \
        "$job")"
    job_id="${raw%%;*}"
    printf 'submitted transfer %s: %s (resume from %s/60)\n' \
        "$name" "$job_id" "$complete"
    submitted=$((submitted + 1))
done

printf 'transfer recovery: submitted %d configs; skipped %d already complete\n' \
    "$submitted" "$skipped"
