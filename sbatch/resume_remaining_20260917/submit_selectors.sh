#!/usr/bin/env bash
# Resume the six incomplete static-selector jobs and two incomplete FUS configs.
set -Eeuo pipefail

ROOT="${ROOT:-/home/mmoslem3/scratch/PoisonBase}"
SOURCE_DIR="$ROOT/sbatch/influence_fus_baselines_20260917"
RESULT_ROOT="${IFB_RESULT_ROOT:-$ROOT/influence_fus_baselines_result}"
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

static_jobs=(
    job_04_loss_gm_b0005_r20.sh
    job_08_loss_sapa_b0005_r20.sh
    job_12_grad_gm_b0005_r20.sh
    job_16_grad_sapa_b0005_r20.sh
    job_20_forget_gm_b0005_r20.sh
    job_24_forget_sapa_b0005_r20.sh
)
fus_jobs=(
    job_28_fus_gm_b0005_r20.sh
    job_32_fus_sapa_b0005_r20.sh
)

submitted_configs=0
submitted_gpu_tasks=0
submitted_merges=0
skipped=0

for name in "${static_jobs[@]}"; do
    job="$SOURCE_DIR/$name"
    [ -f "$job" ] || { printf 'missing job file: %s\n' "$job" >&2; exit 1; }
    run_name="$(sed -n 's/^export IFB_RUN_NAME=//p' "$job")"
    complete="$(python3 "$COUNT" "$RESULT_ROOT/$run_name")"
    if [ "$complete" -ge 60 ]; then
        printf 'skip complete selector config: %s (60/60)\n' "$name"
        skipped=$((skipped + 1))
        continue
    fi

    old_job_name="$(sed -n 's/^#SBATCH --job-name=//p' "$job")"
    resume_job_name="${old_job_name/ifb/ifbr}"
    raw="$(submit_sbatch --parsable \
        --job-name="$resume_job_name" \
        --output="$ROOT/sbatch/logs/${resume_job_name}-%j.out" \
        "$job")"
    job_id="${raw%%;*}"
    printf 'submitted selector %s: %s (resume from %s/60)\n' \
        "$name" "$job_id" "$complete"
    submitted_configs=$((submitted_configs + 1))
    submitted_gpu_tasks=$((submitted_gpu_tasks + 1))
done

# FUS has target partitions of 3, 3, 2, and 2 targets. Submit only partitions
# that are still incomplete, then merge all four persistent part directories.
part_expected=(18 18 12 12)
for name in "${fus_jobs[@]}"; do
    job="$SOURCE_DIR/$name"
    [ -f "$job" ] || { printf 'missing job file: %s\n' "$job" >&2; exit 1; }
    run_name="$(sed -n 's/^export IFB_RUN_NAME=//p' "$job")"
    complete="$(python3 "$COUNT" "$RESULT_ROOT/$run_name")"
    if [ "$complete" -ge 60 ]; then
        printf 'skip complete FUS config: %s (60/60)\n' "$name"
        skipped=$((skipped + 1))
        continue
    fi

    missing_parts=()
    for part_index in 0 1 2 3; do
        part_number=$((part_index + 1))
        part_dir="$RESULT_ROOT/$run_name/parts/part_${part_number}_of_4"
        part_complete="$(python3 "$COUNT" "$part_dir")"
        if [ "$part_complete" -lt "${part_expected[$part_index]}" ]; then
            missing_parts+=("$part_index")
            printf 'FUS %s part %d: %s/%s; will resume\n' \
                "$name" "$part_number" "$part_complete" \
                "${part_expected[$part_index]}"
        fi
    done

    merge_job="$SOURCE_DIR/${name/job_/merge_}"
    [ -f "$merge_job" ] || { printf 'missing merge file: %s\n' "$merge_job" >&2; exit 1; }
    old_job_name="$(sed -n 's/^#SBATCH --job-name=//p' "$job")"
    resume_job_name="${old_job_name/ifb/ifbr}"
    merge_old_name="$(sed -n 's/^#SBATCH --job-name=//p' "$merge_job")"
    merge_resume_name="${merge_old_name/ifbm/ifbrm}"

    if [ "${#missing_parts[@]}" -gt 0 ]; then
        array_spec="$(IFS=,; printf '%s' "${missing_parts[*]}")"
        raw="$(submit_sbatch --parsable \
            --array="$array_spec" \
            --job-name="$resume_job_name" \
            --output="$ROOT/sbatch/logs/${resume_job_name}-%A_%a.out" \
            "$job")"
        job_id="${raw%%;*}"
        merge_raw="$(submit_sbatch --parsable \
            --dependency="afterok:$job_id" \
            --job-name="$merge_resume_name" \
            --output="$ROOT/sbatch/logs/${merge_resume_name}-%j.out" \
            "$merge_job")"
        merge_id="${merge_raw%%;*}"
        printf 'submitted FUS %s: %s array=%s; merge=%s\n' \
            "$name" "$job_id" "$array_spec" "$merge_id"
        submitted_gpu_tasks=$((submitted_gpu_tasks + ${#missing_parts[@]}))
    else
        merge_raw="$(submit_sbatch --parsable \
            --job-name="$merge_resume_name" \
            --output="$ROOT/sbatch/logs/${merge_resume_name}-%j.out" \
            "$merge_job")"
        merge_id="${merge_raw%%;*}"
        printf 'all FUS parts already complete for %s; submitted merge=%s\n' \
            "$name" "$merge_id"
    fi
    submitted_configs=$((submitted_configs + 1))
    submitted_merges=$((submitted_merges + 1))
done

printf 'selector recovery: submitted %d configs as %d GPU tasks plus %d merges; skipped %d complete\n' \
    "$submitted_configs" "$submitted_gpu_tasks" "$submitted_merges" "$skipped"
printf 'Existing stale dependency merge jobs are not cancelled automatically.\n'
