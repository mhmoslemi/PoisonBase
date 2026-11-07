#!/bin/bash
# Usage: sh submit_saved_defenses_vulcan.sh [batch-index ...]
# Re-running submits only unfinished bundles that are not currently queued/running.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -Eeuo pipefail

ROOT="${DEFENSE_ROOT:-/scratch/mmoslem3/PoisonBase}"
BATCH="$ROOT/sbatch/rebuttal_saved_defenses"
OUT="$ROOT/rebuttal_saved_defenses_result"
PERTURB_JOB_ID="${PERTURB_JOB_ID:-2168638}"
SELECTED="${*:-}"
mkdir -p "$BATCH/logs" "$OUT/batches"

for selected in "$@"; do
    if ! [[ "$selected" =~ ^[0-9]+$ ]] || [ "$selected" -gt 35 ]; then
        echo "ERROR: batch indices must be 0 through 35." >&2
        exit 1
    fi
done

exec 9>"$BATCH/submission.lock"
flock -n 9 || { echo 'ERROR: another copy of this submitter is running.' >&2; exit 1; }
STATE="$BATCH/submissions.tsv"
touch "$STATE"
ACTIVE="$BATCH/.active_jobs.$$"
trap 'rm -f "$ACTIVE"' EXIT
squeue -h -u "$USER" -o '%A' > "$ACTIVE"

# Only bundles 7 and 32 use any of the four missing perturbations.
POISON_CACHE="$ROOT/ours_result/CIFAR10_ConvNetBN_gradmatch_random_dog-bird_b0.005_eps8_seed42_K20_ce5_tgt70/poison_cache"
PERTURBS_READY=1
for target in 8954 821 1547 4427; do
    if [ ! -s "$POISON_CACHE/base_${target}.json" ] || [ ! -s "$POISON_CACHE/delta_${target}.pt" ]; then
        PERTURBS_READY=0
    fi
done

submitted=0
while IFS=$'\t' read -r index defense model memory depends targets victims; do
    [ "$index" != index ] || continue
    if [ -n "$SELECTED" ]; then
        case " $SELECTED " in *" $index "*) ;; *) continue ;; esac
    fi
    if [ -s "$OUT/batches/$(printf '%02d' "$index")/done.json" ]; then
        echo "SKIP batch=$index: complete"
        continue
    fi
    last_job=$(awk -F '\t' -v batch_index="$index" '$1 == batch_index {id=$2} END {print id}' "$STATE")
    if [ -n "$last_job" ] && grep -Fxq "$last_job" "$ACTIVE"; then
        echo "SKIP batch=$index: job $last_job is already queued/running"
        continue
    fi
    dependency=()
    dependency_note=none
    if [ "$depends" = 1 ] && [ "$PERTURBS_READY" = 0 ]; then
        dependency=(--dependency="afterok:$PERTURB_JOB_ID")
        dependency_note="afterok:$PERTURB_JOB_ID"
    fi
    job_id=$(sbatch --parsable \
        --account=aip-bghaddar --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --mem="${memory}G" --gres=gpu:l40s:1 --time=06:00:00 \
        --signal=USR1@180 \
        --job-name="reb_${defense}_${index}" \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out" \
        "${dependency[@]}" "$BATCH/worker.sh" "$index")
    job_id="${job_id%%;*}"
    printf '%s\t%s\t%s\t%s\n' "$index" "$job_id" "$dependency_note" "$(date -u +%FT%TZ)" >> "$STATE"
    submitted=$((submitted+1))
    echo "SUBMITTED batch=$index job=$job_id defense=$defense model=$model mem=${memory}G dependency=$dependency_note"
done < "$BATCH/jobs.tsv"
echo "Submitted $submitted jobs. Run this same command again to resume unfinished bundles."
