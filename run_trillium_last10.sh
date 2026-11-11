#!/bin/sh
set -eu
: "${SLURM_JOB_ID:?Run this inside your interactive GPU allocation.}"
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$ROOT"
echo "10 victims: completed trials are skipped; saved checkpoints resume."
trap 'status=$?; if [ "$status" -eq 75 ]; then echo "Checkpoint saved. Rerun sh $0 in a new allocation."; fi' 0
COUNT=0
while read -r CELL TARGET VICTIM; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/10] $CELL: target $TARGET, victim $VICTIM"
    ALLOCATION_END=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
    TASK_SECONDS=$(( $(date -d "$ALLOCATION_END" +%s) - $(date +%s) ))
    if [ "$TASK_SECONDS" -le 180 ]; then echo "Allocation ending. Rerun this script to resume."; exit 75; fi
    bash "$ROOT/experiments/rebuttal_tiny_gm/paired_job.sh" "$ROOT" "$ROOT/sbatch/rebuttal_tiny_gm_vgg13_5x5" "$TASK_SECONDS" "victim_${CELL}_t${TARGET}_v${VICTIM}" < /dev/null
done <<'TRIALS'
bullfrog_dining_b001_basis 7803 2
bullfrog_dining_b001_basis 4951 3
bullfrog_dining_b001_basis 4951 4
bullfrog_dining_b0005_random 9010 1
bullfrog_dining_b0005_basis 6942 0
bullfrog_dining_b0005_basis 6942 2
bullfrog_dining_b0005_basis 6942 4
bullfrog_wagon_b001_random 4709 2
bullfrog_wagon_b001_random 3547 1
bullfrog_wagon_b001_basis 3547 3
TRIALS
echo "All 10 victims complete."
